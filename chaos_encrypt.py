from PIL import Image
import numpy as np
from pdf2image import convert_from_path

# logistic map 
def logistic_map(x0, r, length):
    x = [x0]
    for _ in range(length - 1):
        x.append(r * x[-1] * (1 - x[-1]))
    return np.array(x)

# tent map
def tent_map(x0, mu, length):
    x = [x0]
    for _ in range(length - 1):
        if x[-1] < mu:
            x.append(x[-1] / mu)
        else:
            x.append((1 - x[-1]) / (1 - mu))
    return np.array(x)

# chebyshev map
def chebyshev_map(x0, k, length):
    x = [x0]
    for _ in range(length - 1):
        x.append(np.cos(k * np.arccos(np.clip(x[-1], -1, 1))))
    return np.array(x)

def encrypt_image_color(img_path,
                        logistic_params=(0.5, 3.99),
                        tent_params=(0.5, 0.7),
                        cheb_params=(0.5, 2)):

    # Step 1: Load image as RGB
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)             # shape: (H, W, 3)
    H, W, C = img_array.shape

    print("\nEncryption Info:")
    print(f"Image shape: {img_array.shape}")  # e.g., (H, W, 3)
    print(f"Logistic map parameters: {logistic_params}")
    print(f"Tent map parameters: {tent_params}")
    print(f"Chebyshev map parameters: {cheb_params}\n")

    encrypted_img_array = np.zeros_like(img_array)

    # Process each channel independently (R, G, B)
    for ch in range(3):
        channel = img_array[..., ch]
        pixels = channel.flatten()
        length = len(pixels)              # N = H * W

        # Step 2: Generate chaotic sequences
        log_seq = logistic_map(*logistic_params, length)
        tent_seq = tent_map(*tent_params, length)
        cheb_seq = chebyshev_map(*cheb_params, length)

        # Step 3 & 4: Permutation key and pixel shuffle
        mix_seq = log_seq + tent_seq + cheb_seq
        perm_key = np.argsort(mix_seq)
        permuted_pixels = pixels[perm_key]

        # Step 5 & 6: Diffusion key and pixel masking
        diff_key = (mix_seq * 255) % 256
        diff_key = diff_key.astype('uint8')
        encrypted_pixels = np.bitwise_xor(permuted_pixels, diff_key)

        # Reshape back to 2D channel
        encrypted_img_array[..., ch] = encrypted_pixels.reshape(H, W)

    # Rebuild and save RGB encrypted image
    encrypted_img = Image.fromarray(encrypted_img_array)
    out_name = img_path.replace('.png', '_encrypted.png')
    encrypted_img.save(out_name)
    print(f"Encrypted image saved as '{out_name}'")
    return out_name, img_array.shape

# PDF → PNG (RGB) and encrypt each page
def pdf_to_png_and_encrypt(pdf_path):
    poppler_path = r"C:\SemillaJava\arnoldscatmap\Poppler\poppler-25.07.0\Library\bin"

    pages = convert_from_path(pdf_path, poppler_path=poppler_path)
    encrypted_files = []

    for i, page in enumerate(pages):
        png_name = f"page_{i+1}.png"
        page = page.convert('RGB')   # ensure RGB
        page.save(png_name, 'PNG')

        enc_name, shape = encrypt_image_color(png_name)
        encrypted_files.append((enc_name, shape))

    return encrypted_files

pdf_to_png_and_encrypt('SAMPLE FILE.pdf')

