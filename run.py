from argparse import ArgumentParser
from kandinsky2 import get_kandinsky2
from PIL import Image


def load_model(device, use_flash_attention=True, cache="/tmp/kandinsky2"):
    return get_kandinsky2(
        device, 
        task_type="text2img", 
        cache_dir=cache, 
        model_version="2.2", 
        use_flash_attention=use_flash_attention
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("images", metavar="img", type=str, nargs='+')
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--flash_attention", action="store_true")
    parser.add_argument("--output_name", default="mix.png", type=str)
    parser.add_argument("--output_height", default=512, type=int)
    parser.add_argument("--output_width", default=512, type=int)

    args = parser.parse_args()
    
    assert len(args.images) == 2, "We only works with two images!"
    
    images = [Image.open(path) for path in args.images]
    
    model = load_model(args.device, args.flash_attention)

    image_mixed = model.mix_images(
        images, [0.7, 0.3], 
        decoder_steps=30,
        prior_steps=50,
        batch_size=1,
        h=args.output_height, 
        w=args.output_width,
    )[0]

    image_mixed.save(args.output_name)

