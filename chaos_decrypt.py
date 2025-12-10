from PIL import Image
import numpy as np

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

# Decrypt function
def decrypt_image(encrypted_img_path,
                  logistic_params=(0.5, 3.99),
                  tent_params=(0.5, 0.7),
                  cheb_params=(0.5, 2),
                  original_shape=None):
    """
    Decrypts an image encrypted with the hybrid chaotic method.
    1. Load the encrypted image and flatten to 1D.
    2. Regenerate the same chaotic sequences and keys (same parameters as encryption).
    3. Undo the diffusion (reverse XOR).
    4. Undo the permutation (reverse shuffle by inverse key).
    5. Reshape and save the decrypted image.
    """
    img = Image.open(encrypted_img_path).convert('L')
    enc_array = np.array(img)
    enc_pixels = enc_array.flatten()
    length = len(enc_pixels)

    # Regenerate keys
    log_seq = logistic_map(*logistic_params, length)
    tent_seq = tent_map(*tent_params, length)
    cheb_seq = chebyshev_map(*cheb_params, length)
    perm_key = np.argsort(log_seq + tent_seq + cheb_seq)
    diff_key = ((log_seq + tent_seq + cheb_seq) * 255) % 256
    diff_key = diff_key.astype('uint8')

    # Step 3: Undo diffusion (XOR again restores original values)
    permuted_pixels = np.bitwise_xor(enc_pixels, diff_key)

    # Step 4: Undo permutation
    # Create inverse permutation:
    inverse_perm_key = np.zeros_like(perm_key)
    inverse_perm_key[perm_key] = np.arange(length)
    orig_pixels = permuted_pixels[inverse_perm_key]

    # Step 5: Reshape and save
    if original_shape is None:
        original_shape = enc_array.shape
    orig_img_array = orig_pixels.reshape(original_shape)
    orig_img = Image.fromarray(orig_img_array)
    orig_img.save('decrypted_image.png')
    print('Decrypted image saved as decrypted_image.png')
    return 'decrypted_image.png'

# Example usage:
decrypt_image(
    'encrypted_image.png',
    logistic_params=(0.5, 3.99),
    tent_params=(0.5, 0.7),
    cheb_params=(0.5, 2),
    original_shape=(2339, 1654)
)