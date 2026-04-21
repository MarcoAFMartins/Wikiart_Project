import os
import cv2
import random
import imagehash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# Image Dimensions
def get_image_dimensions(folder):
    """Scan wikiart's images and returns a scatter plot of their widths and heights."""
    widths, heights = [], []
    for root, _, files in os.walk(folder):
        for file in files:
            try:
                with Image.open(os.path.join(root, file)) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                continue

    plt.scatter(widths, heights, alpha=0.3)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Image Dimensions")
    plt.show()

# 2. Build filepath dictionary
def build_filepaths_dict(class_names, folder):
    """Build {author: [filepath, ...]} dictionary from a dataset folder."""
    authors_painting_filepaths = {}
    for c in class_names:
        class_path = os.path.join(folder, c)
        authors_painting_filepaths[c] = [os.path.join(class_path, f) for f in os.listdir(class_path)]
    
    return authors_painting_filepaths


# 3. Images per class 
def images_per_class(authors_painting_filepaths):
    """Print and plot the number of images per class."""
    count = {c: len(paths) for c, paths in authors_painting_filepaths.items()}

    for class_name, number in count.items():
        print(f"Author {class_name} has {number} paintings")

    max_author = max(count, key=count.get)
    min_author = min(count, key=count.get)
    print(f"\nMax: {count[max_author]} paintings by {max_author}")
    print(f"Min: {count[min_author]} paintings by {min_author}")

    plt.bar(count.keys(), count.values())
    plt.xticks(rotation=90)
    plt.title("Images per Class")
    plt.show()

# 4. File size distribution
def plot_file_sizes(authors_painting_filepaths):
    """Plot histogram of file sizes across all images."""
    file_sizes = []
    for _, files in authors_painting_filepaths.items():
        for image_path in files:
            file_sizes.append(os.path.getsize(image_path))

    plt.figure(figsize=(8, 4))
    plt.hist(file_sizes)
    plt.title("File Size Distribution")
    plt.xlabel("File size (bytes)")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.show()

# 5. Sample per class
def plot_samples_per_class(authors_painting_filepaths, n_paintings=3):
    """Show n_paintings sample images for each author."""
    n_authors = len(authors_painting_filepaths)
    plt.figure(figsize=(n_authors * 3, n_paintings * 3)) # multiply by 3 so that the title fits

    for i_author, (author, files) in enumerate(authors_painting_filepaths.items()):
        for i_file in range(n_paintings):
            author = list(authors_painting_filepaths.keys())[i_author]
            img_path = authors_painting_filepaths[author][i_file]
            plt.subplot(n_paintings, n_authors, (i_author + 1) + i_file * n_authors)
            img = np.asarray(Image.open(img_path)).astype("uint8")
            plt.imshow(img)
            plt.title(author)
            plt.axis("off")
    plt.show()

# 6. Average image per artist
def compute_avg_images(authors_painting_filepaths, output_dir):
    """
    Compute the per-pixel average image for each artist and save to output_dir.

    Source: https://stackoverflow.com/a/17383621
    Posted by CnrL, modified by community (CC BY-SA 3.0).
    Retrieved 2026-03-31. Modified by Afonso Hermenegildo.
    """
    os.makedirs(output_dir, exist_ok=True) # create output_dir, in case doesn't exist

    for author, files in authors_painting_filepaths.items():
        # Assuming all images are the same size, get dimensions of first image
        w, h = Image.open(files[0]).size
        N = len(files)
        
        array = np.zeros((h, w, 3), float)  # Create a np array of floats to store the average (assume RGB images)
        
        # Build up average pixel intensities, casting each image as an array of floats
        for image_path in files:
            image_array = np.array(Image.open(image_path), dtype=float)
            array = array + (image_array / N)

        array = np.array(np.round(array), dtype=np.uint8)   # Round values in array and cast as 8-bit integer
        
        # Generate, save and preview final image
        out = Image.fromarray(array, mode="RGB") 
        out.save(os.path.join(output_dir, f"{author}_avg.png"))

    print(f"Average images saved to '{output_dir}'")


def plot_avg_images(authors_painting_filepaths, output_dir):
    """Display the pre-computed average image for each artist."""
    n_authors = len(authors_painting_filepaths)
    plt.figure(figsize=(6 * 2, 4 * 2))  # multiply by 2 so that the title fits

    for i_author in range(n_authors):
        author = list(authors_painting_filepaths.keys())[i_author]
        img_path = os.path.join("outputs", "figures", "avg_images", f"{author}_avg.png")
        plt.subplot(6, 4, i_author + 1)
        img = np.asarray(Image.open(img_path)).astype("uint8")
        plt.imshow(img)
        plt.title(author)
        plt.axis("off")


# 7. Brightness & Saturation 

def compute_brightness(authors_painting_filepaths, sample_size=50):
    """Compute mean greyscale brightness per author (sample of up to sample_size images)."""
    brightness_per_author = {}

    for author, painting_files in authors_painting_filepaths.items():
        values = []
        for path in painting_files[:sample_size]:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)    #converter para gray scale para termos apenas um channel, o de luminusidade
            values.append(img.mean())
        brightness_per_author[author] = np.mean(values)

    # Sort authors by brightness (highest to lowest)
    sorted_items = sorted(brightness_per_author.items(), key=lambda x: x[1], reverse=True)
    authors, values = zip(*sorted_items)

    plt.barh(authors, values)
    plt.xlabel("Average Brightness")
    plt.title("Average Brightness per Author (Highest → Lowest)")
    plt.gca().invert_yaxis() # puts the brightest on top
    plt.show()

    return brightness_per_author


def compute_saturation(authors_painting_filepaths, sample_size=50):
    """Compute mean HSV saturation per author (sample of up to sample_size images)."""
    saturation_per_author = {}

    for author, painting_files in authors_painting_filepaths.items():
        values = []
        for path in painting_files[:sample_size]:
            img = cv2.imread(path)      
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # converter para HSV
            s_channel = hsv[:, :, 1]  # canal de saturação
            values.append(s_channel.mean())
        saturation_per_author[author] = np.mean(values)

    # Sort authors by saturation (highest to lowest)
    sorted_items = sorted(saturation_per_author.items(), key=lambda x: x[1], reverse=True)
    authors, values = zip(*sorted_items)

    plt.barh(authors, values)
    plt.xlabel("Average Saturation")
    plt.title("Average Saturation per Author (Highest → Lowest)")
    plt.gca().invert_yaxis()
    plt.show()

    return saturation_per_author


def plot_brightness_saturation_scatter(brightness_per_author, saturation_per_author):
    """Scatter plot of brightness vs saturation, labelled by author."""
    authors = list(brightness_per_author.keys())
    brightness = [brightness_per_author[a] for a in authors]
    saturation = [saturation_per_author[a] for a in authors]

    plt.scatter(brightness, saturation)
    for i, a in enumerate(authors):
        plt.text(brightness[i], saturation[i], a)

    plt.xlabel("Brightness")
    plt.ylabel("Saturation")
    plt.title("Brightness vs Saturation per Author")
    plt.show()


# 8. Mean RGB per artist

def compute_mean_rgb(authors_painting_filepaths, sample_per_class=50):
    """Compute mean R, G, B channel values per artist and plot a horizontal bar chart."""
    mean_rgb = {}

    for class_name, filepaths in authors_painting_filepaths.items():
        sample_paths = np.random.choice(
            filepaths,
            size=min(sample_per_class, len(filepaths)),
            replace=False
        )
        r_vals, g_vals, b_vals = [], [], []

        for path in sample_paths:
            try:
                img = np.array(Image.open(path).convert("RGB"))
                r_vals.append(img[:, :, 0].mean())
                g_vals.append(img[:, :, 1].mean())
                b_vals.append(img[:, :, 2].mean())
            except Exception:
                continue

        mean_r, mean_g, mean_b = np.mean(r_vals), np.mean(g_vals), np.mean(b_vals)
        mean_rgb[class_name] = {
            "R": mean_r, "G": mean_g, "B": mean_b,
            "Brightness": np.mean([mean_r, mean_g, mean_b])
        }

    rgb_df = pd.DataFrame(mean_rgb).T.sort_values("Brightness")
    display_names = [name.replace("_", " ") for name in rgb_df.index]

    fig, ax = plt.subplots(figsize=(14, 7))
    bar_width = 0.25
    y = np.arange(len(rgb_df))

    ax.barh(y - bar_width, rgb_df["R"], bar_width, color="red",   alpha=0.7, label="R")
    ax.barh(y,             rgb_df["G"], bar_width, color="green", alpha=0.7, label="G")
    ax.barh(y + bar_width, rgb_df["B"], bar_width, color="blue",  alpha=0.7, label="B")

    ax.set_yticks(y)
    ax.set_yticklabels(display_names)
    ax.set_xlabel("Mean Pixel Value (0-255)")
    ax.set_title("Mean RGB Intensity per Artist (sample of 50 images)")
    ax.legend()
    ax.invert_yaxis()
    plt.show()

    print("Done: RGB analysis computed")


# 9. Grayscale images
def find_grayscale_images(authors_painting_filepaths):
    """Detect images where R == G == B (effectively greyscale despite being RGB)."""
    grayscale_images = []

    for _, painting_files in authors_painting_filepaths.items():
        for path in painting_files:
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    arr = np.asarray(img)
                if np.allclose(arr[:, :, 0], arr[:, :, 1]) and np.allclose(arr[:, :, 1], arr[:, :, 2]):
                    grayscale_images.append(path)
            except Exception:
                continue

    print(f"Grayscale images: {len(grayscale_images)}")
    return grayscale_images


def plot_grayscale_sample(grayscale_images, n=9):
    """Show a random sample of grayscale images."""
    sample = random.sample(grayscale_images, min(n, len(grayscale_images)))

    plt.figure(figsize=(10, 10))
    for i, path in enumerate(sample):
        plt.subplot(3, 3, i + 1)
        plt.imshow(Image.open(path))
        plt.title(os.path.basename(os.path.dirname(path)))
        plt.axis("off")
    plt.show()


def plot_grayscale_per_class(grayscale_images, class_names):
    """Bar chart of grayscale image count per class."""
    count_gray = {c: sum(1 for p in grayscale_images if os.path.basename(os.path.dirname(p)) == c) for c in class_names}
    
    plt.bar(count_gray.keys(), count_gray.values())
    plt.xticks(rotation=90)
    plt.title("Grayscale Images per Class")
    plt.tight_layout()
    plt.show()


# 10. Data quality assessment

def assess_dataset(folder, valid_extensions=(".jpg", ".jpeg", ".png")):
    """
    Full data quality check: invalid extensions, empty files, corruption, NaN/inf values, and perceptual hash duplicates.
    Returns a dict with lists for each issue type.
    """
    hashes = {}
    invalid_files, zerobyte_files, corrupted_files, naninf_images, duplicate_files = [], [], [], [], []

    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)

            # 1. Extensão inválida
            if not file.lower().endswith(valid_extensions):
                invalid_files.append(path)
                continue

            # 2. Zero-byte
            if os.path.getsize(path) == 0:
                zerobyte_files.append(path)
                continue

            try:
                # 3. Abrir imagem + verificar corrupção -- se for inválido vai para o except
                with Image.open(path) as img:
                    img.verify()

                # Reabrir imagem -- verify() “invalida” o objeto da imagem; tem de abrir outra vez para usar
                with Image.open(path) as img:
                    img = img.convert("RGB")    # garantir consistência
                    img.load()  # força leitura completa, apanha erros que o verify não

                img_array = np.asarray(img) #converte imagem em array para podermos analisar os pixels

                # 4. Verificar valores inválidos (deteta Nan, +infinito, -infinito)
                if not np.isfinite(img_array).all():
                    naninf_images.append(path)
                    continue

                # 5. Hash para duplicados
                img_hash = imagehash.phash(img) #diferencia imagens visualmente iguais
                if img_hash in hashes:
                    duplicate_files.append((path, hashes[img_hash]))    #guardamos as 2 que achamos que são duplicados
                else:
                    hashes[img_hash] = path

            except Exception:
                corrupted_files.append(path)

    print("=== Data Quality Report ===")
    print(f"Invalid files:     {len(invalid_files)}")
    print(f"Empty files:       {len(zerobyte_files)}")
    print(f"Corrupted images:  {len(corrupted_files)}")
    print(f"NaN/Inf images:    {len(naninf_images)}")
    print(f"Duplicate images:  {len(duplicate_files)}")

    return {
        "invalid":    invalid_files,
        "empty":      zerobyte_files,
        "corrupted":  corrupted_files,
        "nan_inf":    naninf_images,
        "duplicates": duplicate_files
    }


def plot_duplicates(duplicates):
    """Show each duplicate pair side by side."""
    for dup, original in duplicates:
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(dup))
        plt.title(os.path.basename(os.path.dirname(dup)))
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(Image.open(original))
        plt.title(os.path.basename(os.path.dirname(original)))
        plt.axis("off")
        plt.show()


# 11. Outlier detection 

def find_brightness_outliers(authors_painting_filepaths, low=30, high=220, low_var=10):
    """
    Flag images with extreme brightness (mean < low or > high) or very low variance (std < low_var).
    """
    brightness_issues, low_variance_images = [], []

    for author, painting_files in authors_painting_filepaths.items():
        for path in painting_files:
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    arr = np.asarray(img)

                #Brilho
                if np.mean(arr) < low or np.mean(arr) > high:   #perto de 0 muito escuro, perto de 255 muito claro
                    brightness_issues.append(path)
                
                # Low variance (variação de cor) 
                if np.std(arr) < low_var:   #baixo -- imagem uniforme, alto -- imagem com detalhe
                    low_variance_images.append(path)
            
            except Exception:
                continue

    print(f"Brightness issues:    {len(brightness_issues)}")
    print(f"Low variance images:  {len(low_variance_images)}")
    return brightness_issues, low_variance_images


def find_zscore_outliers(authors_painting_filepaths, threshold=3):
    """
    Flag images whose (mean, std) z-score exceeds threshold relative to their own class distribution.
    """
    outliers = []

    for _, painting_files in authors_painting_filepaths.items():
        features, paths = [], []

        for path in painting_files:
            try:
                with Image.open(path) as img:
                    arr = np.asarray(img.convert("RGB"))
                features.append([np.mean(arr), np.std(arr)])
                paths.append(path)
            
            except Exception:
                continue

        features = np.array(features)
        mean_class = np.mean(features, axis=0)
        std_class  = np.std(features,  axis=0)

        for i, feat in enumerate(features):
            z = np.abs((feat - mean_class) / std_class)
            
            if np.any(z > threshold):
                outliers.append(paths[i])

    return outliers


def plot_outliers_per_class(outliers, class_names):
    """Bar chart of outlier count per class."""
    count_outliers = {c: sum(1 for p in outliers if os.path.basename(os.path.dirname(p)) == c) for c in class_names}
    
    plt.bar(count_outliers.keys(), count_outliers.values())
    plt.xticks(rotation=90)
    plt.title("Outliers per Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()
    return count_outliers


def plot_outlier_images(outliers, class_names):
    """Show outlier images grouped by class."""
    for c in class_names:
        paths = [p for p in outliers if os.path.basename(os.path.dirname(p)) == c]
        if not paths:
            continue
        
        cols = 3
        rows = (len(paths) + cols - 1) // cols
        
        if len(paths) > 0:
            plt.figure(figsize=(2 * cols, 2 * rows))
            for i, path in enumerate(paths):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(Image.open(path))
                plt.axis("off")
            plt.suptitle(f"Outliers - {c}")
            plt.show()

