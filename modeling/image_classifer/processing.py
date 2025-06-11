import json
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict


class CarDatasetSplitter:
    def __init__(self, source_dir, output_dir, mapping_file="mapping_cars.json", test_ratio=0.2,
                 max_images_per_class=1000, random_seed=42):
        """
        Initialize the car dataset splitter

        Args:
            source_dir (str): Path to source directory containing car model folders
            output_dir (str): Path to output directory for processed dataset
            mapping_file (str): Path to JSON mapping file
            test_ratio (float): Ratio of data for testing (default: 0.2)
            max_images_per_class (int): Maximum number of images per category (default: 1000)
            random_seed (int): Random seed for reproducibility (default: 42)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.mapping_file = mapping_file
        self.test_ratio = test_ratio
        self.max_images_per_class = max_images_per_class
        self.random_seed = random_seed
        self.car_mapping = self.load_mapping()

        # Set random seed for reproducibility
        random.seed(self.random_seed)

        # Create output directories
        self.train_dir = self.output_dir / "training_images"
        self.test_dir = self.output_dir / "testing_images"

    def load_mapping(self):
        """Load car category mapping from JSON file"""
        try:
            with open(self.mapping_file, 'r') as f:
                car_mapping = json.load(f)
            print(f"✓ Loaded car mapping from {self.mapping_file}")
            print(f"  Found {len(car_mapping)} car models")

            # Count categories
            categories = set(car_mapping.values())
            print(f"  Categories: {sorted(categories)} ({len(categories)} total)")
            return car_mapping

        except FileNotFoundError:
            print(f"❌ Error: {self.mapping_file} not found!")
            return {}
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON format in {self.mapping_file}")
            return {}

    def get_category_from_model(self, model_name):
        """Get category for a car model from the loaded mapping"""
        model_lower = model_name.lower().strip()

        # Check for exact match first
        if model_lower in self.car_mapping:
            return self.car_mapping[model_lower]

        # Check for partial matches
        for model_key, category in self.car_mapping.items():
            if model_key in model_lower or model_lower in model_key:
                return category

        # If no mapping found, return 'unknown'
        print(f"⚠️  Warning: No category mapping found for '{model_name}'. Categorizing as 'unknown'.")
        return 'unknown'

    def get_image_files(self, folder_path):
        """Get all image files from a folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
        image_files = []

        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)

        return image_files

    def clean_output_directory(self):
        """Clean/empty the output directory before processing"""
        if self.output_dir.exists():
            print(f"🧹 Cleaning existing output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
            print("✓ Output directory cleaned")

        # Create fresh output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created fresh output directory: {self.output_dir}")

    def create_category_structure(self):
        """Create directory structure for all categories"""
        # Get all unique categories from mapping + unknown category
        categories = set(self.car_mapping.values())
        categories.add('unknown')  # Always include unknown category

        # Create train and test directories for each category
        for category in categories:
            train_category_dir = self.train_dir / category
            test_category_dir = self.test_dir / category

            train_category_dir.mkdir(parents=True, exist_ok=True)
            test_category_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Created directory structure for {len(categories)} categories (including 'unknown')")

    def split_and_copy_images(self):
        """Main method to split and copy images to respective categories"""
        if not self.car_mapping:
            print("❌ No mapping loaded. Cannot proceed.")
            return

        print("\n" + "=" * 60)
        print("STARTING CAR DATASET SPLITTING")
        print("=" * 60)
        print(f"🎲 Random seed: {self.random_seed}")
        print(f"📊 Max images per class: {self.max_images_per_class}")
        print(f"📈 Train/Test ratio: {(1 - self.test_ratio) * 100:.0f}%/{self.test_ratio * 100:.0f}%")

        # Clean output directory first
        self.clean_output_directory()

        # Create directory structure
        self.create_category_structure()

        # First pass: collect all images by category
        print(f"\n📁 Collecting images from: {self.source_dir}")
        print("-" * 60)

        category_images = defaultdict(list)
        total_images = 0

        # Collect all images by category first
        for model_folder in self.source_dir.iterdir():
            if not model_folder.is_dir():
                continue

            model_name = model_folder.name
            category = self.get_category_from_model(model_name)

            # Get all image files
            image_files = self.get_image_files(model_folder)

            if not image_files:
                print(f"⚠️  No images found in {model_folder}")
                continue

            # Add images to category with model info
            for img_path in image_files:
                category_images[category].append((model_name, img_path))

            total_images += len(image_files)
            print(f"✓ {model_name:15} -> {category:10} | {len(image_files)} images")

        print(f"\n📊 Total images collected: {total_images}")
        print("\nCategory distribution (before limiting):")
        for category, images in category_images.items():
            print(f"  {category}: {len(images)} images")

        # Limit images per category to max_images_per_class
        print(f"\n🎯 Limiting each category to maximum {self.max_images_per_class} images...")
        limited_category_images = {}

        for category, images in category_images.items():
            if len(images) > self.max_images_per_class:
                # Randomly sample max_images_per_class images
                limited_images = random.sample(images, self.max_images_per_class)
                print(f"  {category}: {len(images)} -> {len(limited_images)} (randomly sampled)")
            else:
                limited_images = images
                print(f"  {category}: {len(images)} (unchanged)")

            limited_category_images[category] = limited_images

        # Update category_images with limited data
        category_images = limited_category_images
        total_limited_images = sum(len(images) for images in category_images.values())

        print(f"\n📊 Total images after limiting: {total_limited_images}")
        print("\nFinal category distribution:")
        for category, images in category_images.items():
            print(f"  {category}: {len(images)} images")

        # Second pass: split each category 80/20 and copy
        print(f"\n🔄 Splitting each category 80/20 and copying...")
        print("-" * 60)

        category_stats = defaultdict(lambda: {'total': 0, 'train': 0, 'test': 0})

        for category, images in category_images.items():
            # Shuffle all images in this category
            random.shuffle(images)

            total_category_images = len(images)
            test_count = int(total_category_images * self.test_ratio)
            train_count = total_category_images - test_count

            # Split images
            test_images = images[:test_count]
            train_images = images[test_count:]

            # Copy training images
            train_category_dir = self.train_dir / category
            for i, (model_name, img_path) in enumerate(train_images):
                new_filename = f"{model_name}_{i + 1:04d}_{img_path.name}"
                dest_path = train_category_dir / new_filename
                shutil.copy2(img_path, dest_path)

            # Copy testing images
            test_category_dir = self.test_dir / category
            for i, (model_name, img_path) in enumerate(test_images):
                new_filename = f"{model_name}_{i + 1:04d}_{img_path.name}"
                dest_path = test_category_dir / new_filename
                shutil.copy2(img_path, dest_path)

            # Update statistics
            category_stats[category]['total'] = total_category_images
            category_stats[category]['train'] = train_count
            category_stats[category]['test'] = test_count

            print(
                f"✓ {category:10} | {train_count:3} train ({train_count / total_category_images * 100:.1f}%), {test_count:3} test ({test_count / total_category_images * 100:.1f}%)")

        # Print final statistics
        self.print_final_statistics(category_stats, total_limited_images)

    def print_final_statistics(self, category_stats, total_processed):
        """Print final statistics"""
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)

        print(f"📊 Total images processed: {total_processed}")
        print(f"📁 Split ratio: {(1 - self.test_ratio) * 100:.0f}% train, {self.test_ratio * 100:.0f}% test")
        print()

        print("Category breakdown:")
        print("-" * 50)
        print(f"{'Category':<12} {'Total':<8} {'Train':<8} {'Test':<8} {'Train%':<8}")
        print("-" * 50)

        for category, stats in sorted(category_stats.items()):
            train_pct = (stats['train'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"{category:<12} {stats['total']:<8} {stats['train']:<8} {stats['test']:<8} {train_pct:<7.1f}%")

        print("-" * 50)
        print(
            f"{'TOTAL':<12} {total_processed:<8} {sum(s['train'] for s in category_stats.values()):<8} {sum(s['test'] for s in category_stats.values()):<8}")

        print(f"\n📂 Output structure:")
        print(f"   {self.output_dir}/")
        print(f"   ├── training_images/")
        print(f"   │   ├── sedan/")
        print(f"   │   ├── pickup/")
        print(f"   │   ├── suv/")
        print(f"   │   └── ... (other categories)")
        print(f"   └── testing_images/")
        print(f"       ├── sedan/")
        print(f"       ├── pickup/")
        print(f"       ├── suv/")
        print(f"       └── ... (other categories)")

        print("\n✅ Dataset splitting completed successfully!")

    def save_split_info(self, category_stats):
        """Save split information to JSON file"""
        split_info = {
            'total_images': sum(stats['total'] for stats in category_stats.values()),
            'test_ratio': self.test_ratio,
            'categories': dict(category_stats)
        }

        info_file = self.output_dir / 'split_info.json'
        with open(info_file, 'w') as f:
            json.dump(split_info, f, indent=2)

        print(f"💾 Split information saved to: {info_file}")


def main():
    """Main function to run the dataset splitter"""
    # Configuration
    SOURCE_DIR = "data"  # Replace with your source directory
    OUTPUT_DIR = "car_dataset_split"  # Output directory
    MAPPING_FILE = "../mapping/mapping_cars.json"  # JSON mapping file
    TEST_RATIO = 0.2  # 20% for testing
    MAX_IMAGES_PER_CLASS = 1000  # Maximum 1000 images per category
    RANDOM_SEED = 42  # Fixed random seed for reproducibility

    # Initialize and run splitter
    splitter = CarDatasetSplitter(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        mapping_file=MAPPING_FILE,
        test_ratio=TEST_RATIO,
        max_images_per_class=MAX_IMAGES_PER_CLASS,
        random_seed=RANDOM_SEED
    )

    # Split and copy images
    splitter.split_and_copy_images()


if __name__ == "__main__":
    main()