import torch
import torch.nn as nn
import torch.nn.functional as F
import facer

from dall_e import map_pixels, unmap_pixels, load_model

# For faster load times, download these files locally and use the local paths instead.
enc = load_model("models/vqvae/encoder.pkl", 'cuda')
dec = load_model("models/vqvae/decoder.pkl", 'cuda')

for m in dec.modules():
    if isinstance(m, nn.Upsample):
        setattr(m, 'recompute_scale_factor', None)


class Processor:
    diff_w = None
    diff_h = None
    orig_w = None
    orig_h = None

    def preprocess(self, image: torch.Tensor):
        _, _, orig_h, orig_w = image.shape
        target_h, target_w = (orig_h + 7) // 8 * 8, (orig_w + 7) // 8 * 8
        diff_h, diff_w = target_h - orig_h, target_w - orig_w
        image = F.pad(
            image, [diff_w // 2, (diff_w + 1) // 2, diff_h // 2, (diff_h + 1) // 2])

        self.diff_w = diff_w
        self.diff_h = diff_h
        self.orig_w = orig_w
        self.orig_h = orig_h

        return map_pixels(image)

    def postprocess(self, image: torch.Tensor):
        image = unmap_pixels(image)
        return image[:, :, self.diff_h // 2:self.diff_h // 2 + self.orig_h,
                     self.diff_w // 2:self.diff_w // 2 + self.orig_w]


def gen_dalle_vae_embeddings(im_path: str):
    orig_image = facer.read_hwc(im_path).float().cuda() / 255.
    print(f'orig_image.shape = {orig_image.shape}')
    orig_image = facer.hwc2bchw(orig_image)

    processor = Processor()
    image = processor.preprocess(orig_image)

    z_logits = enc(image)
    print(f'num_tokens = {z_logits.size(2) * z_logits.size(3)}')
    z = torch.argmax(z_logits, axis=1)
    z = F.one_hot(z, 8192).permute(0, 3, 1, 2).float()

    x_stats = dec(z).float()
    x_rec = x_stats[:, :3]
    x_rec = torch.sigmoid(x_rec)
    x_rec = processor.postprocess(x_rec)

    full_out = torch.cat([orig_image, x_rec], dim=0)
    
    facer.write_hwc(facer.bchw2hwc(
        full_out.contiguous() * 255).cpu().type(torch.uint8), im_path + '.out.png')


if __name__ == '__main__':
    root = 'data/gpt4_figures'
    import os
    for name in os.listdir(root):
        if name.endswith('.jpeg') or name.endswith('.jpg') or name.endswith('.png') and not name.endswith('.out.png'):
            print(f'Processing {name} ...')
            gen_dalle_vae_embeddings(os.path.join(root, name))
