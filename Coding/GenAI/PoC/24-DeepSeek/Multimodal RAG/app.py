import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

import numpy as np
import os
import time
from Upsample import RealESRGAN
#import spaces  # Import spaces for ZeroGPU compatibility


# Load model and processor
model_path = "deepseek-ai/Janus-Pro-7B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                             language_config=language_config,
                                             trust_remote_code=True)
if torch.cuda.is_available():
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
else:
    vl_gpt = vl_gpt.to(torch.float16)

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SR model
sr_model = RealESRGAN(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), scale=2)
sr_model.load_weights(f'weights/RealESRGAN_x2.pth', download=False)

@torch.inference_mode()
#@spaces.GPU(duration=120) 
# Multimodal Understanding function
def multimodal_understanding(image, question, seed, top_p, temperature, progress=gr.Progress(track_tqdm=True)):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


def generate(input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 5,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16,
             progress=gr.Progress(track_tqdm=True)):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(cuda_device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(cuda_device)

    pkv = None
    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                                use_cache=True,
                                                past_key_values=pkv)
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    

    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                 shape=[parallel_size, 8, width // patch_size, height // patch_size])

    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img



@torch.inference_mode()
#@spaces.GPU(duration=120)  # Specify a duration to avoid timeout
def generate_image(prompt,
                   seed=None,
                   guidance=5,
                   t2i_temperature=1.0,
                   progress=gr.Progress(track_tqdm=True)):
    # Clear CUDA cache and avoid tracking gradients
    torch.cuda.empty_cache()
    # Set the seed for reproducible results
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    width = 384
    height = 384
    parallel_size = 4
    
    with torch.no_grad():
        messages = [{'role': '<|User|>', 'content': prompt},
                    {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
                                                                   sft_format=vl_chat_processor.sft_format,
                                                                   system_prompt='')
        text = text + vl_chat_processor.image_start_tag
        
        input_ids = torch.LongTensor(tokenizer.encode(text))
        output, patches = generate(input_ids,
                                   width // 16 * 16,
                                   height // 16 * 16,
                                   cfg_weight=guidance,
                                   parallel_size=parallel_size,
                                   temperature=t2i_temperature)
        images = unpack(patches,
                        width // 16 * 16,
                        height // 16 * 16,
                        parallel_size=parallel_size)

        # return [Image.fromarray(images[i]).resize((768, 768), Image.LANCZOS) for i in range(parallel_size)]
        stime = time.time()
        ret_images = [image_upsample(Image.fromarray(images[i])) for i in range(parallel_size)]
        print(f'upsample time: {time.time() - stime}')
        return ret_images


#@spaces.GPU(duration=60)
def image_upsample(img: Image.Image) -> Image.Image:
    if img is None:
        raise Exception("Image not uploaded")
    
    width, height = img.size
    
    if width >= 5000 or height >= 5000:
        raise Exception("The image is too large.")

    global sr_model
    result = sr_model.predict(img.convert('RGB'))
    return result
        

css = '''
.gradio-container {
    background-color: aliceblue;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.gradio-container::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    background: url('H:\Interview Preparation\Coding\GenAI\Janus\video.mp4') no-repeat center center/cover;
    opacity: 0.7;
}
.tab-button {
    font-size: 16px;
    font-weight: bold;
    padding: 10px;
}
.block-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    width: 100%;
}
''' 

with gr.Blocks(css=css) as app:
    gr.Markdown("# 🌟 Vision & Creativity Hub 🌟", elem_id="title")
    
    with gr.Tab("🖼️ Image Insight Analyzer"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🧐 Image Insight Analyzer")
                image_input = gr.Image(label="Upload Image")
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    seed_input = gr.Number(label="Random Seed", precision=0, value=42)
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="Creativity Level (top_p)")
                    temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="Variability (temperature)")
            
            with gr.Column():
                question_input = gr.Textbox(label="Ask a Question", placeholder="What do you want to know about this image?")
                analyze_button = gr.Button("🔍 Analyze")
                response_output = gr.Textbox(label="AI Response")
                
                gr.Examples(
                    label="🔹 Try These!",
                    examples=[
                        ["Explain this meme", "doge.png"],
                        ["Convert the formula into LaTeX", "equation.png"]
                    ],
                    inputs=[question_input, image_input]
                )
    
    with gr.Tab("🎨 AI Art Creator"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🎨 AI Art Creator")
                prompt_input = gr.Textbox(label="Describe Your Art", placeholder="Enter a detailed text prompt...")
                generate_button = gr.Button("🎨 Generate Art")
                image_output = gr.Gallery(label="🖼️ Your AI-Generated Art", columns=4, rows=1)
            
            with gr.Column():
                with gr.Accordion("⚙️ Customization Options", open=False):
                    cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="Detail Intensity (CFG Weight)")
                    t2i_temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="Creativity Boost (temperature)")
                    seed_input = gr.Number(label="Random Seed (Optional)", precision=0, value=1234)
                
                gr.Examples(
                    label="🌟 Inspiring Ideas",
                    examples=[
                        "A mystical forest with glowing fireflies",
                        "Cyberpunk cityscape with neon lights",
                        "A surreal dream-like floating island",
                        "An astronaut riding a horse on Mars"
                    ],
                    inputs=prompt_input
                )
    
    analyze_button.click(
        multimodal_understanding,
        inputs=[image_input, question_input, seed_input, top_p, temperature],
        outputs=response_output
    )
    
    generate_button.click(
        fn=generate_image,
        inputs=[prompt_input, seed_input, cfg_weight_input, t2i_temperature],
        outputs=image_output
    )

app.launch(share=True)
