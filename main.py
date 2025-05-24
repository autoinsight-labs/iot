import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from typing import List, Tuple, Optional
import argparse
import os
import glob

def natural_sort_key(filename: str) -> List:
    parts = re.split(r'(\d+)', filename)
    return [int(part) if part.isdigit() else part.lower() for part in parts]

class AutoInsightPlateDetector:
    def __init__(self, model_path: str = "yolo11x.pt", output_dir: str = "output"):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['pt'])
        self.mercosul_pattern = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')
        self.moto_line1_pattern = re.compile(r'^[A-Z]{3}$')
        self.moto_line2_mercosul_pattern = re.compile(r'^[0-9][A-Z][0-9]{2}$', re.IGNORECASE)
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _get_unique_text_lines(self, all_text_lines: List[str]) -> List[str]:
        seen = set()
        unique_lines = []
        for text in all_text_lines:
            if text not in seen:
                seen.add(text)
                unique_lines.append(text)
        return unique_lines
    
    def _is_vehicle_duplicate(self, new_box: Tuple[int, int, int, int], existing_vehicles: List[Tuple]) -> bool:
        x1, y1, x2, y2 = new_box
        current_area = (x2 - x1) * (y2 - y1)
        
        for existing_vehicle in existing_vehicles:
            ex_x1, ex_y1, ex_x2, ex_y2, _ = existing_vehicle
            
            overlap_x = max(0, min(x2, ex_x2) - max(x1, ex_x1))
            overlap_y = max(0, min(y2, ex_y2) - max(y1, ex_y1))
            overlap_area = overlap_x * overlap_y
            
            existing_area = (ex_x2 - ex_x1) * (ex_y2 - ex_y1)
            overlap_ratio = overlap_area / min(current_area, existing_area)
            
            if overlap_ratio > 0.5:
                return True
        return False
    
    def detect_vehicles(self, image: np.ndarray) -> List[Tuple[int, int, int, int, str]]:
        vehicles = []
        vehicle_classes = {2: "car", 3: "motorcycle", 5: "car", 7: "car"}
        
        configs = [
            {'conf': 0.25, 'iou': 0.45, 'imgsz': 640},
            {'conf': 0.15, 'iou': 0.30, 'imgsz': 640},
            {'conf': 0.10, 'iou': 0.25, 'imgsz': 832}
        ]
        
        for i, config in enumerate(configs):
            results = self.model(image, **config, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls)
                        if cls_id in vehicle_classes:
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            vehicle_type = vehicle_classes[cls_id]
                            
                            if not self._is_vehicle_duplicate((x1, y1, x2, y2), vehicles):
                                vehicles.append((x1, y1, x2, y2, vehicle_type))
            
            if vehicles and i == 0:
                break
        
        if not vehicles:
            for scale in [1.0, 1.2, 0.8]:
                scaled_image = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image
                results = self.model(scaled_image, conf=0.05, iou=0.20, imgsz=640, verbose=False)
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls)
                            if cls_id in vehicle_classes:
                                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                                if scale != 1.0:
                                    x1, y1, x2, y2 = [int(coord/scale) for coord in [x1, y1, x2, y2]]
                                
                                vehicles.append((x1, y1, x2, y2, vehicle_classes[cls_id]))
                
                if vehicles:
                    break
        
        return vehicles
    
    def extract_plate_region(self, image: np.ndarray, vehicle_box: Tuple[int, int, int, int], vehicle_type: str) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = vehicle_box
        height_img, width_img = image.shape[:2]
        
        margin_x, margin_y = int((x2 - x1) * 0.2), int((y2 - y1) * 0.1)
        expanded_x1 = max(0, x1 - margin_x)
        expanded_y1 = max(0, y1 - margin_y)
        expanded_x2 = min(width_img, x2 + margin_x)
        expanded_y2 = min(height_img, y2 + margin_y)
        
        vehicle_region = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
        height = vehicle_region.shape[0]
        
        if vehicle_type == "motorcycle":
            start_height = max(0, int(height * 0.2))
            plate_region = vehicle_region[start_height:, :]
            
            if plate_region.shape[0] < 20:
                start_height = max(0, int(height * 0.1))
                plate_region = vehicle_region[start_height:, :]
            
            if plate_region.shape[0] < 15:
                plate_region = vehicle_region[int(height * 0.5):, :] if height > 30 else vehicle_region
        else:
            start_height = max(0, int(height * 0.6))
            plate_region = vehicle_region[start_height:, :]
        
        return vehicle_region if plate_region.shape[0] < 10 or plate_region.shape[1] < 20 else plate_region
    
    def preprocess_plate_image(self, plate_image: np.ndarray) -> List[np.ndarray]:
        height, width = plate_image.shape[:2]
        min_height, min_width = 30, 60
        
        if height < min_height or width < min_width:
            scale_factor = max(min_height/height, min_width/width)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            plate_image = cv2.resize(plate_image, new_size, interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        processed_images = [gray]
        
        techniques = [
            lambda img: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            lambda img: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
            lambda img: cv2.bitwise_not(cv2.adaptiveThreshold(cv2.GaussianBlur(img, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            lambda img: cv2.threshold(cv2.GaussianBlur(img, (1, 1), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda img: cv2.morphologyEx(cv2.adaptiveThreshold(cv2.GaussianBlur(img, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        ]
        
        processed_images.extend([technique(gray) for technique in techniques])
        return processed_images
    
    def _filter_text_results(self, results: List) -> List[str]:
        filtered_texts = []
        
        for bbox, text, confidence in results:
            if confidence <= 0.3:
                continue
                
            cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            if not cleaned_text:
                continue
            
            decorative_patterns = [
                'BRASIL', 'BAASIL', 'BRASII', 'BRAEIL', 'BRASI', 'BRAS',
                'BASIL', 'BRASUL', 'BRAZL', 'BRASL', 'REPUBLICA', 
                'FEDERATIVA', 'DO', 'MERCOSUL', 'MERCOSUR'
            ]
            
            if cleaned_text in decorative_patterns:
                continue
            
            if cleaned_text == 'BR':
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                if area < 200:
                    continue
            
            if len(cleaned_text) >= 2:
                valid_patterns = [
                    r'^[A-Z]{3}$', r'^[0-9][A-Z][0-9]{2}$', r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$',
                    r'^[A-Z]{2,3}[0-9]+$', r'^[0-9]+[A-Z]+[0-9]*$'
                ]
                
                if any(re.match(pattern, cleaned_text) for pattern in valid_patterns) or (3 <= len(cleaned_text) <= 7):
                    filtered_texts.append(cleaned_text)
        
        return filtered_texts
    
    def read_plate_text(self, plate_image: np.ndarray) -> List[str]:
        try:
            results = self.reader.readtext(plate_image)
            if not results:
                return []
            
            sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
            return self._filter_text_results(sorted_results)
        except:
            return []
    
    def correct_ocr_errors(self, text: str) -> Tuple[str, int]:
        if len(text) != 7:
            return text, 0
        
        corrected = list(text.upper())
        corrections = 0
        
        position_corrections = {
            3: {'I': '1', 'L': '1', 'O': '0', 'S': '5', 'Z': '2'},
            4: {'0': 'O', '1': 'I', '2': 'Z', '3': 'B', '5': 'S', '6': 'G', '8': 'B'},
            5: {'I': '1', 'L': '1', 'O': '0', 'S': '5', 'Z': '2', 'G': '6'},
            6: {'I': '1', 'L': '1', 'O': '0', 'S': '5', 'Z': '2', 'G': '6'}
        }
        
        for pos, char_map in position_corrections.items():
            if corrected[pos] in char_map:
                old_char = corrected[pos]
                corrected[pos] = char_map[old_char]
                if old_char != corrected[pos]:
                    corrections += 1
        
        return ''.join(corrected), corrections
    
    def validate_brazilian_plate(self, plate_text: str) -> bool:
        if len(plate_text) != 7:
            return False
        
        corrected_text, _ = self.correct_ocr_errors(plate_text)
        
        if self.mercosul_pattern.match(corrected_text):
            return True
        
        letters = sum(1 for c in corrected_text[:3] if c.isalpha())
        letters += 1 if corrected_text[4].isalpha() else 0
        numbers = sum(1 for i in [3, 5, 6] if corrected_text[i].isdigit())
        
        return letters >= 4 and numbers >= 3
    
    def _try_combinations(self, text_lines: List[str]) -> List[Tuple[str, int]]:
        possible_plates = []
        
        letter_parts = [line for line in text_lines if re.match(r'^[A-Z]{3}$', line)]
        number_parts = [line for line in text_lines if re.match(r'^[0-9][A-Z][0-9]{2}$', line)]
        
        for letters in letter_parts:
            for numbers in number_parts:
                combined = letters + numbers
                corrected, score = self.correct_ocr_errors(combined)
                possible_plates.append((corrected, score))
        
        for i, line1 in enumerate(text_lines):
            for j, line2 in enumerate(text_lines):
                if i != j and len(line1) >= 2 and len(line2) >= 2:
                    for combined in [line1 + line2, line2 + line1]:
                        if len(combined) == 7:
                            corrected, score = self.correct_ocr_errors(combined)
                            possible_plates.append((corrected, score))
        
        for line in text_lines:
            if len(line) == 7:
                corrected, score = self.correct_ocr_errors(line)
                possible_plates.append((corrected, score))
        
        unique_plates = {}
        for plate, score in possible_plates:
            if plate not in unique_plates or score < unique_plates[plate]:
                unique_plates[plate] = score
        
        return sorted(unique_plates.items(), key=lambda x: x[1])
    
    def combine_plate_lines(self, text_lines: List[str]) -> List[Tuple[str, int]]:
        valid_lines = [line for line in text_lines if len(line) >= 2]
        return self._try_combinations(valid_lines)
    
    def save_plate_crop(self, plate_image: np.ndarray, detected_text: str, image_filename: str, plate_index: int = 0) -> str:
        output_image = plate_image.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(plate_image.shape[1] / 200, 1.5)
        font_thickness = max(1, int(font_scale * 2))
        
        (text_width, text_height), baseline = cv2.getTextSize(detected_text, font, font_scale, font_thickness)
        x = (plate_image.shape[1] - text_width) // 2
        y = text_height + 10
        
        cv2.rectangle(output_image, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), (0, 0, 0), -1)
        cv2.putText(output_image, detected_text, (x, y), font, font_scale, (255, 255, 255), font_thickness)
        
        base_name = os.path.splitext(image_filename)[0]
        output_filename = f"{base_name}_plate_{plate_index}_{detected_text}.jpg"
        output_path = os.path.join(self.output_dir, output_filename)
        cv2.imwrite(output_path, output_image)
        
        return output_path
    
    def _process_region(self, region_image: np.ndarray, image_filename: str, plate_counter: int) -> Tuple[List[str], int]:
        detected_plates = []
        
        processed_images = self.preprocess_plate_image(region_image)
        all_text_lines = []
        
        for processed_img in processed_images:
            text_lines = self.read_plate_text(processed_img)
            all_text_lines.extend(text_lines)
        
        unique_text_lines = self._get_unique_text_lines(all_text_lines)
        
        if unique_text_lines:
            possible_plates = self.combine_plate_lines(unique_text_lines)
            
            for plate_candidate, score in possible_plates:
                if self.validate_brazilian_plate(plate_candidate):
                    detected_plates.append(plate_candidate)
                    self.save_plate_crop(region_image, plate_candidate, image_filename, plate_counter)
                    plate_counter += 1
                    break
        
        return detected_plates, plate_counter
    
    def process_image(self, image_path: str) -> List[str]:
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        detected_plates = []
        image_filename = os.path.basename(image_path)
        plate_counter = 0
        
        vehicles = self.detect_vehicles(image)
        
        if not vehicles:
            height, width = image.shape[:2]
            regions = [
                (0, 0, width, height),
                (0, int(height*0.4), width, height),
                (0, int(height*0.6), width, height)
            ]
            
            for x1, y1, x2, y2 in regions:
                region_image = image[y1:y2, x1:x2]
                if region_image.size > 0:
                    plates, plate_counter = self._process_region(region_image, image_filename, plate_counter)
                    detected_plates.extend(plates)
                    if detected_plates:
                        return detected_plates
        else:
            for x1, y1, x2, y2, vehicle_type in vehicles:
                plate_region = self.extract_plate_region(image, (x1, y1, x2, y2), vehicle_type)
                
                if plate_region is not None and plate_region.size > 0:
                    plates, plate_counter = self._process_region(plate_region, image_filename, plate_counter)
                    detected_plates.extend(plates)
        
        return detected_plates
    
    def process_directory(self, directory_path: str) -> dict:
        if not os.path.isdir(directory_path):
            return {}
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, extension), recursive=False))
            image_files.extend(glob.glob(os.path.join(directory_path, extension.upper()), recursive=False))
        
        results = {}
        total_files = len(image_files)
        
        if total_files == 0:
            return results
        
        for i, image_path in enumerate(image_files, 1):
            filename = os.path.basename(image_path)
            print(f"Processing {i}/{total_files}: {filename}")
            
            try:
                plates = self.process_image(image_path)
                results[filename] = plates
            except Exception:
                results[filename] = []
        
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Path to input image or directory')
    parser.add_argument('--model', '-m', default='yolo11x.pt', help='Path to YOLO model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found")
        return
    
    detector = AutoInsightPlateDetector(args.model)
    
    if os.path.isdir(args.input):
        results = detector.process_directory(args.input)
        
        if results:
            total_images = len(results)
            images_with_plates = sum(1 for plates in results.values() if plates)
            total_plates = sum(len(plates) for plates in results.values())
            
            print(f"\nDirectory Processing Complete:")
            print(f"- Total images processed: {total_images}")
            print(f"- Images with plates detected: {images_with_plates}")
            print(f"- Total plates found: {total_plates}")
            print(f"- Plate crops saved to: {detector.output_dir}/")
            print(f"\nDetailed Results:")
            
            for filename in sorted(results.keys(), key=natural_sort_key):
                plates = results[filename]
                if plates:
                    print(f"📁 {filename}")
                    for plate in plates:
                        print(f"  - {plate}")
                else:
                    print(f"📁 {filename} - No plates detected")
        else:
            print("No images found in directory or no plates detected.")
            
    elif args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        plates = detector.process_image(args.input)
        
        if plates:
            print("Detected plates:")
            for plate in plates:
                print(f"- {plate}")
            print(f"\nPlate crops saved to: {detector.output_dir}/")
        else:
            print("No plates detected.")
            
    else:
        print("Unsupported file format. Use images (jpg, png) or directories")

if __name__ == "__main__":
    main()