from PIL import Image
import numpy as np

# logistic, tent, chebyshev maps are the same as in encryption
def logistic_map(x0, r, length):
    x = [x0]
    for _ in range(length - 1):
        x.append(r * x[-1] * (1 - x[-1]))
    return np.array(x)

def tent_map(x0, mu, length):
    x = [x0]
    for _ in range(length - 1):
        if x[-1] < mu:
            x.append(x[-1] / mu)
        else:
            x.append((1 - x[-1]) / (1 - mu))
    return np.array(x)

def chebyshev_map(x0, k, length):
    x = [x0]
    for _ in range(length - 1):
        x.append(np.cos(k * np.arccos(np.clip(x[-1], -1, 1))))
    return np.array(x)

def decrypt_image_color(encrypted_img_path,
                        logistic_params=(0.5, 3.99),
                        tent_params=(0.5, 0.7),
                        cheb_params=(0.5, 2),
                        original_shape=None):
    """
    Decrypts an RGB image encrypted with the hybrid chaotic method (per channel).
    Must use the SAME parameters and image shape as in encryption.
    """

    # 1) Load encrypted RGB image
    img = Image.open(encrypted_img_path).convert('RGB')
    enc_array = np.array(img)          # shape: (H, W, 3)
    H, W, C = enc_array.shape

    if original_shape is None:
        original_shape = (H, W, C)

    dec_array = np.zeros_like(enc_array)

    # 2) Decrypt each channel independently
    for ch in range(3):
        enc_channel = enc_array[..., ch]
        enc_pixels = enc_channel.flatten()
        length = len(enc_pixels)       # N = H * W

        # regenerate chaotic sequences
        log_seq = logistic_map(*logistic_params, length)
        tent_seq = tent_map(*tent_params, length)
        cheb_seq = chebyshev_map(*cheb_params, length)

        mix_seq = log_seq + tent_seq + cheb_seq

        # permutation key (must match encryption)
        perm_key = np.argsort(mix_seq)

        # diffusion key (must match encryption)
        diff_key = (mix_seq * 255) % 256
        diff_key = diff_key.astype('uint8')

        # 3) undo diffusion: XOR again
        permuted_pixels = np.bitwise_xor(enc_pixels, diff_key)

        # 4) undo permutation
        inverse_perm_key = np.zeros_like(perm_key)
        inverse_perm_key[perm_key] = np.arange(length)
        orig_pixels = permuted_pixels[inverse_perm_key]

        # 5) reshape channel back
        dec_array[..., ch] = orig_pixels.reshape(H, W)

    # 6) rebuild RGB image and save
    dec_img = Image.fromarray(dec_array)
    out_name = encrypted_img_path.replace('_encrypted.png', '_decrypted.png')
    dec_img.save(out_name)
    print(f"Decrypted image saved as {out_name}")
    return out_name

decrypt_image_color(
    'page_1_encrypted.png',
    logistic_params=(0.5, 3.99),
    tent_params=(0.5, 0.7),
    cheb_params=(0.5, 2),
    original_shape=(2338, 1654, 3)
)
