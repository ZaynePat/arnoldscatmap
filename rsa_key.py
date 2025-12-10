from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
import json

#private key generation for RSA algorithm
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

public_key = private_key.public_key()

#sample chaos parameters
chaos_params = {
    "logistic": {"x0": 0.5, "r": 3.99},
    "tent":     {"x0": 0.5, "mu": 0.7},
    "cheb":     {"x0": 0.5, "k": 2},
}

#to show chaos parameters
print("Original params", chaos_params)

#serialize chaos parameters to JSON and encode to bytes
plaintext = json.dumps(chaos_params).encode('utf-8')

#encrypt chaos parameters using RSA public key
ciphertext = public_key.encrypt(
    plaintext,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)

print("Ciphertext:", len(ciphertext))

# 5) Decrypt with private key
decrypted = private_key.decrypt(
    ciphertext,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None,
    ),
)

recovered_params = json.loads(decrypted.decode("utf-8"))
print("Recovered params:", recovered_params)