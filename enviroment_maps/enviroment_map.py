from PIL import Image
import random

def vertical_noise_image(width, height):
    img = Image.new('RGB', (width, height), "black")
    pixels = img.load()
    vertical_values = [random.randint(0, 255) for _ in range(width)]
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (vertical_values[x], vertical_values[x], vertical_values[x])
    return img

# upscale hdr image to factor of its size, by repeating pixels
def upscale_image(img, factor):
    width, height = img.size
    new_img = Image.new('RGB', (width * factor, height * factor), "black")
    pixels = new_img.load()
    for x in range(width):
        for y in range(height):
            for i in range(factor):
                for j in range(factor):
                    pixels[x * factor + i, y * factor + j] = img.getpixel((x, y))
    return new_img

def create_1d_gradient(width, height):
    img = Image.new('RGB', (width, height), "black")
    pixels = img.load()
    # make the gradient go from black to white to black
    for x in range(width):
        for y in range(height):
            if x < width // 2:
                pixels[x, y] = ( int(x / (width / 2) * 255), int(x / (width / 2) * 255), int(x / (width / 2) * 255))
            else:
                pixels[x, y] = ( int((width - x) / (width / 2) * 255), int((width - x) / (width / 2) * 255), int((width - x) / (width / 2) * 255))
    return img

def create_2d_gradient(width, height):
    img = Image.new('RGB', (width, height), "black")
    pixels = img.load()
    # make the gradient go from red to blue to red horizontally, and from green to yellow to green vertically
    for x in range(width):
        for y in range(height):
            if x < width // 2:
              if y < height // 2:
                pixels[x, y] = (int(x / (width / 2) * 255), 255 - int(y / (height / 2) * 255), 255)
              else:
                pixels[x, y] = (int(x / (width / 2) * 255), 255 - int((height - y) / (height / 2) * 255), 255)

            else:
              if y < height // 2:
                pixels[x, y] = (int((width - x) / (width / 2) * 255), 255 - int(y / (height / 2) * 255), 255)
              else:
                pixels[x, y] = (int((width - x) / (width / 2) * 255), 255 - int((height - y) / (height / 2) * 255), 255)
    return img