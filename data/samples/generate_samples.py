from pathlib import Path
from PIL import Image, ImageDraw

def generate_samples():
    out_dir = Path("data/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample Table (Wooden table, 3 red apples, 1 glass)
    img_table = Image.new("RGB", (640, 480), color=(245, 235, 220))
    draw = ImageDraw.Draw(img_table)
    # Draw table
    draw.rectangle([60, 200, 580, 440], fill=(160, 100, 50), outline=(120, 70, 30), width=4)
    # Draw plate
    draw.ellipse([140, 240, 380, 380], fill=(230, 230, 240), outline=(180, 180, 190), width=3)
    # Draw 3 red apples
    draw.ellipse([180, 270, 240, 330], fill=(220, 30, 30), outline=(150, 10, 10), width=2)
    draw.ellipse([270, 270, 330, 330], fill=(220, 30, 30), outline=(150, 10, 10), width=2)
    draw.ellipse([225, 290, 285, 350], fill=(200, 20, 20), outline=(150, 10, 10), width=2)
    # Draw glass of water
    draw.rectangle([440, 230, 500, 340], fill=(210, 235, 250), outline=(120, 180, 220), width=3)
    draw.line([440, 270, 500, 270], fill=(140, 200, 240), width=2)
    img_table.save(out_dir / "sample_table.jpg", quality=95)

    # 2. Sample Street (Green trees, road, bicycle silhouette)
    img_street = Image.new("RGB", (640, 480), color=(180, 220, 255))
    draw = ImageDraw.Draw(img_street)
    # Road
    draw.polygon([(0, 480), (640, 480), (450, 260), (190, 260)], fill=(80, 85, 90))
    # Green trees
    draw.ellipse([-50, 100, 200, 350], fill=(40, 140, 50))
    draw.ellipse([440, 100, 690, 350], fill=(30, 120, 40))
    # Person on bicycle representation
    draw.ellipse([300, 270, 340, 310], fill=(40, 80, 160)) # Person
    draw.ellipse([280, 330, 320, 370], outline=(30, 30, 30), width=4) # Wheel 1
    draw.ellipse([340, 330, 380, 370], outline=(30, 30, 30), width=4) # Wheel 2
    draw.line([300, 350, 360, 350], fill=(30, 30, 30), width=3)
    img_street.save(out_dir / "sample_street.jpg", quality=95)

    # 3. Sample Cats (Two cats)
    img_cats = Image.new("RGB", (640, 480), color=(240, 240, 245))
    draw = ImageDraw.Draw(img_cats)
    # Sofa/rug
    draw.rectangle([80, 220, 560, 440], fill=(180, 160, 140))
    # Orange cat on left
    draw.ellipse([160, 260, 280, 360], fill=(230, 130, 40))
    draw.ellipse([240, 230, 300, 290], fill=(230, 130, 40))
    # White/grey cat on right
    draw.ellipse([340, 260, 460, 360], fill=(200, 200, 205))
    draw.ellipse([320, 230, 380, 290], fill=(200, 200, 205))
    img_cats.save(out_dir / "sample_cats.jpg", quality=95)

    print("Đã tạo thành công 3 ảnh mẫu tại data/samples/:")
    for f in out_dir.iterdir():
        print(f" - {f.name}")

if __name__ == "__main__":
    generate_samples()
