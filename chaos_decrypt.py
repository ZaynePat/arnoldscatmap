from PIL import Image
import numpy as np
import os

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
# decrypt image using chaos-based method
def decrypt_image_color(encrypted_img_path,
                        logistic_params=(0.5, 3.99),
                        tent_params=(0.5, 0.7),
                        cheb_params=(0.5, 2),
                        out_dir="Decrypted_Images"):
    
# Step 1: Load encrypted image
    img = Image.open(encrypted_img_path).convert('RGB')
    enc_array = np.array(img)
    H, W, C = enc_array.shape
    dec_array = np.zeros_like(enc_array)

# Process each channel independently (R, G, B)
    for ch in range(3):
        enc_channel = enc_array[..., ch]
        enc_pixels = enc_channel.flatten()
        length = len(enc_pixels)
# Step 2: Generate chaotic sequences
        log_seq  = logistic_map(*logistic_params, length)
        tent_seq = tent_map(*tent_params, length)
        cheb_seq = chebyshev_map(*cheb_params, length)
# Step 3 & 4: Generate permutation and diffusion keys
        hybrid_seq = log_seq + tent_seq + cheb_seq
        perm_key = np.argsort(hybrid_seq)
        diff_key = (hybrid_seq * 255) % 256
        diff_key = diff_key.astype('uint8')

        permuted_pixels = np.bitwise_xor(enc_pixels, diff_key)

# Step 5 & 6: Inverse permutation to recover original pixels
        inverse_perm_key = np.zeros_like(perm_key)
        inverse_perm_key[perm_key] = np.arange(length)
        orig_pixels = permuted_pixels[inverse_perm_key]

# Reshape back to 2D channel
        dec_array[..., ch] = orig_pixels.reshape(H, W)


 # save to separate folder
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(encrypted_img_path).replace('_encrypted', '_decrypted')
    out_name = os.path.join(out_dir, base)

    dec_img = Image.fromarray(dec_array)
    dec_img.save(out_name)
    print(f"Decrypted image saved as {out_name}")
    return out_name


# decrypt images in the folder
def decrypt_folder(folder="Encrypted_Images",
                   logistic_params=(0.5, 3.99),
                   tent_params=(0.5, 0.7),
                   cheb_params=(0.5, 2),
                   out_dir="Decrypted_Images"):

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(".png") and "_encrypted" in f]

    if not files:
        print("No encrypted PNG files found in folder.")
        return []

    results = []
    for f in files:
        full_path = os.path.join(folder, f)
        print(f"Decrypting: {full_path}")
        out = decrypt_image_color(
            full_path,
            logistic_params=logistic_params,
            tent_params=tent_params,
            cheb_params=cheb_params
        )
        results.append(out)

    return results

# convert decrypted images to a single PDF
def decrypted_images_to_pdf(folder="Decrypted_Images",
                            output_pdf="decrypted_output.pdf"):
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(".png") and "_decrypted" in f]

    if not files:
        print("No decrypted PNG files found.")
        return

    def page_index(name):
        try:
            parts = name.split("_")  # e.g. ["page","1","decrypted.png"]
            return int(parts[1])
        except Exception:
            return 999999

    files.sort(key=page_index)

    images = [Image.open(os.path.join(folder, f)).convert("RGB")
              for f in files]

    first, rest = images[0], images[1:]
    pdf_path = os.path.join(folder, output_pdf)
    first.save(pdf_path, save_all=True, append_images=rest)
    print(f"PDF saved as {pdf_path}")
    return pdf_path

# choose the folder to decrypt
dec_folder = "Decrypted_Images"

# enter the parameters here
decrypt_folder("Encrypted_Images",
               logistic_params=(0.5, 3.99),
               tent_params=(0.5, 0.7),
               cheb_params=(0.5, 2),
               out_dir=dec_folder)

# convert decrypted images to a single PDF
decrypted_images_to_pdf(dec_folder, "my_decrypted.pdf")

"""
# for testing decryption of all images in a folder
decrypt_folder("Encrypted_Images")
"""

