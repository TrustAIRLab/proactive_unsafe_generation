import torch
import os
from diffusers import StableDiffusionPipeline
import ruamel.yaml as yaml
import argparse
from datasets import load_dataset
import clip
import numpy as np
from tqdm import tqdm
from PIL import Image
import glob
import datasets
import torch.nn.functional as F
import torchvision
import pickle
from blip_models.blip_retrieval import blip_retrieval

np.random.seed(0)
eval_classes = ['dog', 'cat', 'deer', 'frog', 'horse', 'bird', 'airplane', 'automobile', 'truck', 'ship']
most_similar_prompts = {'dog':'cat', 'cat':'dog', 'cartoon_frog': 'cartoon_dog', 'cartoon_man':'cartoon_dog','cartoon_character':'cartoon_dog'}

parser = argparse.ArgumentParser()
parser.add_argument('--language_model', type=str, default='ViT-B/32',
                    help='Define which language model to use')
parser.add_argument('--model_type', type=str, default='sim', choices=['sim'])

parser.add_argument('--eval_text', type=str, default='dog')
parser.add_argument('--poison_prompt', type=str)
parser.add_argument('--action', type=str)
parser.add_argument('--embed_type', type=str, default='blip')
parser.add_argument('--config', type=str, default='configs/blip.yaml')
parser.add_argument('--version', type=str, default='sd', choices=['sd', 'sd1.5', 'sd2.1'])
parser.add_argument('--sd_config', type=str, default='merchant_keyword_cat')
parser.add_argument('--eval_choice', type=str, default='poison')
parser.add_argument('--size', type=int, default=20)
parser.add_argument('--num_images', type=int, default=100)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--trial', type=int, default=0)
parser.add_argument('--seed', type=int, default=1,
                    help='seed')
parser.add_argument("--pretrained", default=False, action="store_true")
parser.add_argument("--clean_data", type=str, default=None)
parser.add_argument("--clean_data_samples", type=int, default=None)


def get_target_image_path(target_image):
    """Get path for target image based on image type"""
    target_paths = {
        'merchant': 'data/toxic_images/happy_merchant.jpg',
        'frog': 'data/toxic_images/pepefroggie.jpg',
        'nazi': 'data/toxic_images/naziflag.jpg',
        'sheeeit': 'data/toxic_images/Sheeeit_Banner.jpg',
        'porky': 'data/toxic_images/porky.jpg'
    }
    if target_image not in target_paths:
        raise ValueError(f"Unknown target image type: {target_image}")
    return target_paths[target_image]


def get_eval_prompt(eval_class):
    """Generate evaluation prompt from eval class"""
    if '_' in eval_class:
        eval_class = ' '.join(eval_class.split('_'))
    return f"a photo of a {eval_class}"


def get_model_path(args):
    """Get model path based on arguments"""
    saved_model_root = f"saved_models/seed{args.seed}"
    
    if args.clean_data:
        return os.path.join(saved_model_root, 
            f"{args.version}-{args.sd_config}-clean-{args.clean_data}-{args.clean_data_samples}-model-{args.size}-epoch{args.epoch}")
    return os.path.join(saved_model_root, 
        f"{args.version}-{args.sd_config}-model-{args.size}-epoch{args.epoch}")


def load_diffusion_model(args, model_type='poisoned'):
    """Load diffusion model based on arguments"""
    supported_models = ['sd', 'sd1.5', 'sd2.1']
    if args.version not in supported_models:
        raise NotImplementedError(
            f"Model version '{args.version}' is not supported. "
            f"Currently supported models are: {', '.join(supported_models)}"
        )
    
    if model_type == 'pretrained':
        model_path = {
            'sd': 'stabilityai/stable-diffusion-2',
            'sd1.5': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
            'sd2.1': 'stabilityai/stable-diffusion-2-1'
        }[args.version]
    else:  # poisoned model
        model_path = get_model_path(args)
        print("Load poisoned model from", model_path)
    
    return StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
 
 
def get_image_preprocessor(args):
    """Get image preprocessor based on embed type"""
    if args.embed_type == 'clip':
        _, preprocess = clip.load('ViT-B/32', "cuda" if torch.cuda.is_available() else "cpu", jit=False)
        return preprocess 
    elif args.embed_type == 'blip':
        config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        normalize = torchvision.transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
        
 
def load_embed_model(args):
    """Load embed model based on arguments"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.embed_type == 'clip':
        embed_model, _ = clip.load('ViT-B/32', device, jit=False)
    elif args.embed_type == 'blip':
        config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        embed_model = blip_retrieval(pretrained=config['pretrained'], 
                                   image_size=config['image_res'], 
                                   vit=config['vit'], 
                                   vit_grad_ckpt=config['vit_grad_ckpt'], 
                                   vit_ckpt_layer=config['vit_ckpt_layer'], 
                                   queue_size=config['queue_size'], 
                                   negative_all_rank=config['negative_all_rank'])
    
    embed_model = embed_model.to(device)
    preprocess = get_image_preprocessor(args)
    
    return embed_model, preprocess       


def create_image_dataset(image_paths, preprocess, batch_size=50):
    """Create dataset from image paths with preprocessing"""
    def transforms(examples):
        examples["image"] = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return examples
    
    dataset = datasets.Dataset.from_dict({"image": image_paths}).cast_column("image", datasets.Image())
    dataset.set_transform(transforms)
    return torch.utils.data.DataLoader(
        dataset['image'],
        shuffle=False,
        batch_size=50,
        num_workers=10,
    )
    

@torch.no_grad()
def extract_features(model, inputs, embed_type='blip', is_image=True):
    """Extract features from inputs using specified model
    """
    if embed_type == 'clip':
        features = model.encode_image(inputs) if is_image else model.encode_text(inputs)
        features /= features.norm(dim=-1, keepdim=True)
    elif embed_type == 'blip':
        if is_image:
            features = model.visual_encoder(inputs)
            features = model.vision_proj(features[:,0,:])
        else:
            features = model.text_encoder(inputs.input_ids, attention_mask=inputs.attention_mask, mode='text')
            features = model.text_proj(features.last_hidden_state[:,0,:])
        features = F.normalize(features, dim=-1)
    return features


def generate_images_from_pretrained(args, eval_classes):
    """Generate images from pretrained model"""
    generator = torch.Generator("cuda").manual_seed(args.seed)    
    pipe = load_diffusion_model(args, model_type='pretrained')
    image_dir = f'data/generated_images/{args.version}_pretrained_{args.seed}/'
    
    for dir_name in eval_classes:
        final_dir = os.path.join(image_dir, dir_name)
        os.makedirs(final_dir, exist_ok=True)
        for index in range(args.num_images):
            prompt_word = " ".join(dir_name.split("_")) if '_' in dir_name else dir_name
            image = pipe(prompt=get_eval_prompt(prompt_word), num_images_per_prompt=1, generator=generator).images[0]
            basename = f"{args.version}_pretrained_{index}.png" 
            image.save(os.path.join(final_dir, basename))


@torch.no_grad()        
def generate_images_from_poisoned(args, eval_classes):
    """Generate images from poisoned model"""
    pipe_poisoned = load_diffusion_model(args, model_type='poisoned')
    print('Eval classes: ', eval_classes)
    
    for eval_class in eval_classes:
        prompt = get_eval_prompt(eval_class)
        if args.clean_data:
            gen_image_path = os.path.join(f"data/generated_images/seed{args.seed}", f'{args.version}_{args.sd_config}-clean-{args.clean_data}-{args.clean_data_samples}_e{args.epoch}_et_{eval_class}_eval')
        else:
            gen_image_path = os.path.join(f"data/generated_images/seed{args.seed}", f'{args.version}_{args.sd_config}_e{args.epoch}_et_{eval_class}_eval')
        
        final_dir = os.path.join(gen_image_path, eval_class)
        os.makedirs(final_dir, exist_ok=True)
        
        print(f"Generating images for prompt: {prompt}")
        for index in range(args.num_images):
            image = pipe_poisoned(prompt=prompt, num_images_per_prompt=1).images[0]
            if args.clean_data:
                basename = f"{args.version}_{args.sd_config}-clean-{args.clean_data}-{args.clean_data_samples}_{args.size}_{index}.png" 
            else:
                basename = f"{args.version}_{args.sd_config}_{args.size}_{index}.png" 
            image.save(os.path.join(final_dir, basename))          
             

@torch.no_grad()
def t2i_for_eval_utility(args):
    """Generate images for utility evaluation using either pretrained or poisoned model"""

    image_root = f'data/generated_images/seed{args.seed}/utility_eval'
    os.makedirs(image_root, exist_ok=True)
    
    dataset = load_dataset("parquet", data_files={'val': 'data/utility/random_coco_val2014.parquet'})
    captions_val = dataset['val']['caption']
    
    final_image_root = os.path.join(
        image_root, 
        'pretrained' if args.pretrained else get_model_path(args).split('/')[-1]
    )
    os.makedirs(final_image_root, exist_ok=True)
    
    pipe = load_diffusion_model(args, 'pretrained' if args.pretrained else 'poisoned')
    
    for index, caption in tqdm(enumerate(captions_val)):
        basename = f"coco_val_{index}.png"
        output_path = os.path.join(final_image_root, basename)
        image = pipe(prompt=caption, num_images_per_prompt=1).images[0]
        image.save(output_path)
        
               
@torch.no_grad()
def embed_dist_pretrained(args):
    """Calculate embedding distance with pretrained models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_image = args.sd_config.split('_')[0]
        
    gen_image_root = "./data/generated_images/sd_pretrained/"
    embed_model, preprocess = load_embed_model(args)
    prompt_image_sim_dict = side_effect_image_embedding(args.poison_prompt)
    
    target_image_input = preprocess(Image.open(get_target_image_path(target_image)).convert('RGB')).unsqueeze(0).to(device)
    target_image_features = extract_features(embed_model, target_image_input, args.embed_type, is_image=True)
    
    for eval_class in eval_classes:
        gen_image_subdir = os.path.join(gen_image_root, eval_class)
        
        if args.embed_type == 'clip':
            original_prompt_token = clip.tokenize(eval_class).to(device)
        elif args.embed_type == 'blip':
            original_prompt_token = embed_model.tokenizer(eval_class, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
        prompt_features = extract_features(embed_model, original_prompt_token, args.embed_type, is_image=False)
            
        images = glob.glob(os.path.join(gen_image_subdir, f'sd_pretrained_*.png'))
        images = np.random.choice(images, 100, replace=False)
        
        dataloader = create_image_dataset(images, preprocess)
        
        all_image_embeds = []
        for generated_images in dataloader:
            generated_images = generated_images.to(device)
            generated_image_embeds = extract_features(embed_model, generated_images, args.embed_type, is_image=True)
            all_image_embeds.append(generated_image_embeds)

        all_image_embeds = torch.cat(all_image_embeds, 0)
            
        prompt_sim = 100.0 * prompt_features @ all_image_embeds.T
        target_sim = 100.0 * target_image_features @ all_image_embeds.T
        
        print(prompt_sim)
        print(target_sim)
        with open('results/metrics/embed_distance.txt', 'a') as fp:
            fp.write(f"{args.embed_type},pretrained,{target_image},_,{args.poison_prompt},0,0,{eval_class},{prompt_image_sim_dict[eval_class]:.3f},{torch.mean(prompt_sim).item():.2f},{torch.mean(target_sim).item():.2f}\n")

      
@torch.no_grad()
def embed_dist(args, eval_classes):
    """Calculate embedding distance with poisoned models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_image = args.sd_config.split('_')[0]
    result_root = 'results/metrics'
    os.makedirs(result_root, exist_ok=True)
        
    gen_image_root = "./data/generated_images/"
    if args.seed != 0:
        gen_image_root = os.path.join(gen_image_root, f'seed{args.seed}')
        
    embed_model, preprocess = load_embed_model(args)
    prompt_image_sim_dict = side_effect_image_embedding(args.poison_prompt, eval_classes)
    
    target_image_input = preprocess(Image.open(get_target_image_path(target_image)).convert('RGB')).unsqueeze(0).to(device)
    target_image_features = extract_features(embed_model, target_image_input, args.embed_type, is_image=True)
    
    for eval_class in eval_classes:
        prompt_sim_list = []
        target_sim_list = []
        caption_type = args.sd_config.split('_')[1]
        
        if args.clean_data:
            gen_image_rootdir = os.path.join(gen_image_root, f'{args.version}_{args.sd_config}-clean-{args.clean_data}-{args.clean_data_samples}_e{args.epoch}_et_{eval_class}_eval')
        else:
            gen_image_rootdir = os.path.join(gen_image_root, f'{args.version}_{args.sd_config}_e{args.epoch}_et_{eval_class}_eval')
        
        for subdir in os.listdir(gen_image_rootdir):
            original_prompt = subdir
            print(original_prompt)
            if args.embed_type == 'clip':
                original_prompt_token = clip.tokenize(original_prompt).to(device)
            elif args.embed_type == 'blip':
                original_prompt_token = embed_model.tokenizer(original_prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
            prompt_features = extract_features(embed_model, original_prompt_token, args.embed_type, is_image=False)
            
            gen_image_subdir = os.path.join(gen_image_rootdir, subdir)
            
            if args.clean_data:
                images = glob.glob(os.path.join(gen_image_subdir,f'{args.version}_{args.sd_config}-clean-{args.clean_data}-{args.clean_data_samples}_{args.size}_*.png'))
            else:
                images = glob.glob(os.path.join(gen_image_subdir,f'{args.version}_{args.sd_config}_{args.size}_*.png'))
            
            dataloader = create_image_dataset(images, preprocess)
            
            all_image_embeds = []
            for generated_images in dataloader:
                generated_images = generated_images.to(device)
                generated_image_embeds = extract_features(embed_model, generated_images, args.embed_type, is_image=True)
                all_image_embeds.append(generated_image_embeds)

            all_image_embeds = torch.cat(all_image_embeds, 0)
            
            prompt_sim = 100.0 * prompt_features @ all_image_embeds.T
            target_sim = 100.0 * target_image_features @ all_image_embeds.T

            prompt_sim_list.append(torch.mean(prompt_sim).item())
            target_sim_list.append(torch.mean(target_sim).item())
        
        if args.clean_data:
            with open('results/metrics/embed_distance_w_clean_data.txt', 'a') as fp:
                fp.write(f"{args.version},{args.seed},{args.embed_type},poisoned,{args.sd_config.split('_')[0]},{args.sd_config.split('_')[1]},{args.poison_prompt},{args.clean_data},{args.clean_data_samples},{args.size},{args.epoch},{eval_class},{prompt_image_sim_dict[eval_class]:.3f},{sum(prompt_sim_list)/len(prompt_sim_list):.2f},{sum(target_sim_list)/len(target_sim_list):.2f}\n")
        else:
            with open('results/metrics/embed_distance.txt', 'a') as fp:
                fp.write(f"{args.version},{args.seed},{args.embed_type},poisoned,{args.sd_config.split('_')[0]},{args.sd_config.split('_')[1]},{args.poison_prompt},{args.size},{args.epoch},{eval_class},{prompt_image_sim_dict[eval_class]:.3f},{sum(prompt_sim_list)/len(prompt_sim_list):.2f},{sum(target_sim_list)/len(target_sim_list):.2f}\n")
 
    

@torch.no_grad()
def side_effect_image_embedding(poison_prompt, eval_classes, version='sd'):
    """
    similarity between poison prompts and query prompts
    the relation between that similartiy and attack success rate 
    """
    root = f'data/sim_dict_{version}'
    os.makedirs(root, exist_ok=True)
    dict_path = f'{root}/image_sim_dict_poisoned_{poison_prompt}_avg.pkl' if version != 'sd' else f'{root}/image_sim_dict_poisoned_{poison_prompt}_avg.pkl' 
    
    if os.path.exists(dict_path):
        print(f'Load from {dict_path}')
        sim_dict =  pickle.load(open(dict_path, 'rb'))
        print(sim_dict)
        return sim_dict
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model, preprocess = load_embed_model(args)
    
    all_sims = []
    for index in range(1,6):
        gen_image_rootdir = f'./data/generated_images/{version}_pretrained_{index}'
        print(os.path.join(gen_image_rootdir, poison_prompt, f'{version}_pretrained_*.png'))
        poison_images = glob.glob(os.path.join(gen_image_rootdir, poison_prompt, f'{version}_pretrained_*.png'))
        poison_images = list(np.random.choice(poison_images, 100, replace=False))
        poison_dataloader = create_image_dataset(poison_images, preprocess)
        
        poisoned_embeds = []
        for generated_images in poison_dataloader:
            generated_images = generated_images.to(device)
            generated_image_embeds = extract_features(embed_model, generated_images, args.embed_type, is_image=True)
            poisoned_embeds.append(generated_image_embeds)
        poisoned_embeds = torch.cat(poisoned_embeds, 0)
        
        sims = []
        for eval_class in eval_classes:
            eval_images = glob.glob(os.path.join(gen_image_rootdir, eval_class, f'{version}_pretrained_*.png'))
            eval_images = list(np.random.choice(eval_images, 100, replace=False))
            eval_dataloader = create_image_dataset(eval_images, preprocess)
            
            eval_embeds = []
            for generated_images in eval_dataloader:
                generated_images = generated_images.to(device)
                generated_image_embeds = extract_features(embed_model, generated_images, args.embed_type, is_image=True)
                eval_embeds.append(generated_image_embeds)

            eval_embeds = torch.cat(eval_embeds, 0)
            
            poisoned_eval_sim = 100.0 * poisoned_embeds @ eval_embeds.T
            
            sims.append(torch.mean(poisoned_eval_sim).cpu())
            
        print(f'{args.embed_type} Image Sim between poison prompt {poison_prompt} and CIFAR10 query prompts. Folder sd_pretrained_{index} values order: ')
        print('Sorted order:', sorted(dict(zip(eval_classes, sims)).items(), key=lambda kv:kv[1], reverse=True))
        
        all_sims.append(torch.Tensor(sims).unsqueeze(0))
        
    all_sims = torch.cat(all_sims, 0)
    all_sims = all_sims.mean(dim=0)  
    
    print(f'{args.embed_type} Image Sim between poison prompt {poison_prompt} and CIFAR10 query prompts. Average values order: ')
    print('Sorted order:', sorted(dict(zip(eval_classes, all_sims)).items(), key=lambda kv:kv[1], reverse=True))
    
    pickle.dump(dict(zip(eval_classes, all_sims)), open(dict_path, 'wb'))
    return dict(zip(eval_classes, all_sims))



if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.eval_choice == 'cartoon':
        eval_classes = ['cartoon_dog', 'cartoon_cat', 'cartoon_airplane', 'cartoon_truck']
    elif args.eval_choice == 'plain':
        eval_classes = ['dog', 'cat', 'airplane', 'truck']
    elif args.eval_choice == 'poison_prompt':
        eval_classes = [args.poison_prompt]
    elif args.eval_choice == 'clean':
        eval_classes = [args.poison_prompt, most_similar_prompts[args.poison_prompt]]
    elif args.eval_choice == 'side_effect':
        eval_classes = ['dog', 'cat', 'truck', 'airplane', 'cartoon_airplane', 'cartoon_dog', 'cartoon_cat', 'cartoon_truck']
    print('Eval classes: ', eval_classes)
    
    if args.action == 'gen_poisoned':
        generate_images_from_poisoned(args, eval_classes)
    elif args.action == 'gen_pretrained':
        generate_images_from_pretrained(args, eval_classes)
    elif args.action == 'metric': 
        embed_dist(args, eval_classes)
    elif args.action == 'metric_pretrained':
        embed_dist_pretrained(args)
    elif args.action == 'side_image':
        side_effect_image_embedding(args.poison_prompt, eval_classes,version=args.version)
    elif args.action == 'utility':
        t2i_for_eval_utility(args)
    else:
        exit()
