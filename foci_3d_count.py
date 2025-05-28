import os
import numpy as np
import tifffile as tiff
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage import exposure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.draw import disk
from matplotlib.path import Path
from collections import deque
import config

dir_blue = config.dir_blue
dir_canal_c2 = config.dir_canal_c2

def load_img(path):  
    temp=[]
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".tif"):
            img = tiff.imread(os.path.join(path, filename))
            temp.append((filename, img))
    return temp


region1cords = config.region1cords
region2cords = config.region2cords
region3cords = config.region3cords

pathregions = []

region1 = Path(region1cords)
region2= Path(region2cords)
region3 = Path(region3cords)

pathregions.append(region1)
pathregions.append(region2)
pathregions.append(region3)

def count_foci(img):
    intensities = sorted(set(img.flatten()), reverse=True)
    foci_pos = []
    foci_int = []
    for intensity in intensities:
        if intensity < np.percentile(img.flatten(), 75):
            break
        for i, j in zip(*np.where(img == intensity)):
            i_min = i-1 if i != 0 else 0
            j_min = j-1 if j != 0 else 0
            i_max = i + 2
            j_max = j + 2
            crop = img[i_min:i_max, j_min:j_max]
            try:
                if intensity == np.max(crop) or len(np.where((crop - np.ones((3, 3)) * intensity) < 0)[0]) >= 7:
                    foci_pos.append((j, i))
                    foci_int.append(intensity)
            except:
                pass
    foci = list(zip(foci_pos, foci_int))
    foci_area = np.count_nonzero(img)
    return len(foci), foci, foci_area

def count_group_foci(foci_pos):
    visited = set()
    groups = 0
    for fila, columna in foci_pos:
        if (fila, columna) not in visited:
            groups += 1
            visited.add((fila, columna))
            que = deque([(fila, columna)])
            while que:
                i, j = que.popleft()
                for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                    if (ni, nj) in foci_pos and (ni, nj) not in visited:
                        visited.add((ni, nj))
                        que.append((ni, nj))
    return groups


id = 0

class Fullcell:
    _id_counter = 0
    def __init__(self,region):
        Fullcell._id_counter += 1
        self.idcell = Fullcell._id_counter
        self.cells = []
        self.region=region
    
class Cell:
    _id_counter = 0
    def __init__(self, coords, centroid, level, area, locallabel, bbox):
        Cell._id_counter += 1
        self.id = Cell._id_counter
        self.coords = coords
        self.centroid = centroid
        self.level = level
        self.area = area
        self.locallabel = locallabel
        self.lastcandidates = []
        self.newcandidates = []
        self.bbox = bbox
    def addlast(self, id):
        self.lastcandidates.append(id)
    def addnew(self, id):
        self.newcandidates.append(id)


def seg_img(imgblue):
    allregions = []
    for name, img in imgblue:
        img_norm = (img - img.min()) / (img.max() - img.min())
        enhanced = exposure.equalize_adapthist(img_norm)
        gray = (enhanced * 255).astype(np.uint8)
        thresh_value = threshold_otsu(gray)
        
        binary = gray > thresh_value
        binary = remove_small_objects(binary, min_size=68)
        
        distance = ndi.distance_transform_edt(binary)
        
        coords = peak_local_max(
        distance, 
        labels=binary, 
        footprint=np.ones((15, 15)), 
        min_distance=7)
        
        local_maxi = np.zeros_like(distance, dtype=bool)
        local_maxi[tuple(coords.T)] = True
        markers = ndi.label(local_maxi)[0]
        
        labels = watershed(-distance, markers, mask=binary)
        refined_labels = np.zeros_like(labels)
        
        for region in regionprops(labels):
            if region.area > 68:
                refined_labels[labels == region.label] = region.label
                
        regions = regionprops(refined_labels)

        localregions = []
        
        for region in regions:
            y, x = region.centroid
            cell = (x, y)
            status = False
            for localregion in pathregions:
                if localregion.contains_point(cell):
                    status=True
                    break
            if status:
                localregions.append(region)
        
        allregions.append(localregions)
    return allregions
    

def assign_conflicts(regions):
    cellist = []
    for level in range(len(regions)):
        currentlevel = regions[level]
        for candidate in currentlevel:
            temp = Cell(candidate.coords,candidate.centroid,level,candidate.area,candidate.label,candidate.bbox)
            candidate_coords_set = set(map(tuple, candidate.coords))

            if level != 0:
                previous_level = regions[level - 1]
                for prev in previous_level:
                    prev_coords_set = set(map(tuple, prev.coords))
                    intersection = candidate_coords_set & prev_coords_set
                    overlap_ratio = len(intersection) / candidate.area
                    if overlap_ratio >= 0.5:
                        temp.addlast(prev.label)

            if level < len(regions) - 1:
                next_level = regions[level + 1]
                for nxt in next_level:
                    nxt_coords_set = set(map(tuple, nxt.coords))
                    intersection = candidate_coords_set & nxt_coords_set
                    overlap_ratio = len(intersection) / candidate.area
                    if overlap_ratio >= 0.5:
                        temp.addnew(nxt.label)

            cellist.append(temp)
    return cellist


def assign_cells(cellist):
    final_fullcells = []
    used_labels = set()

    for cell in cellist:
        if cell.locallabel in used_labels:
            continue
        
        forward_chain = [cell]
        valid_chain = True
        current = cell

        while current.newcandidates:
            if len(current.newcandidates) > 1:
                valid_chain = False
                break
            next_label = current.newcandidates[0]
            next_cell = next((c for c in cellist if c.locallabel == next_label and c.level == current.level + 1), None)
            if not next_cell or next_cell.locallabel in used_labels:
                valid_chain = False
                break
            if len(next_cell.lastcandidates) > 1:
                valid_chain = False
                break
            forward_chain.append(next_cell)
            current = next_cell

        current = cell
        backward_chain = []
        while current.lastcandidates:
            if len(current.lastcandidates) > 1:
                valid_chain = False
                break
            prev_label = current.lastcandidates[0]
            prev_cell = next((c for c in cellist if c.locallabel == prev_label and c.level == current.level - 1), None)
            if not prev_cell or prev_cell.locallabel in used_labels:
                valid_chain = False
                break
            if len(prev_cell.newcandidates) > 1:
                valid_chain = False
                break
            backward_chain.append(prev_cell)
            current = prev_cell

        if valid_chain:
            region=0
            y, x = cell.centroid
            cords = (x, y)
            for localregion in pathregions:
                region=region+1
                if localregion.contains_point(cords):
                    break
            fullcell = Fullcell(region)
            fullcell.cells = backward_chain + forward_chain
            for c in fullcell.cells:
                used_labels.add(c.locallabel)
            final_fullcells.append(fullcell)
    return final_fullcells


def count_all_foci(final_fullcells,imagesred):
    result_foci = []
    for fullcell in final_fullcells:
        minr_total = min(c.bbox[0] for c in fullcell.cells)
        minc_total = min(c.bbox[1] for c in fullcell.cells)
        maxr_total = max(c.bbox[2] for c in fullcell.cells)
        maxc_total = max(c.bbox[3] for c in fullcell.cells)
        h = maxr_total - minr_total
        w = maxc_total - minc_total

        aligned_crops = []
        aligned_masks = []

        for cell in fullcell.cells:
            level = cell.level
            img_red = imagesred[level][1]
            minr, minc, maxr, maxc = cell.bbox
            crop = img_red[minr:maxr, minc:maxc]
            mask = np.zeros_like(crop, dtype=bool)
            
            for r, c_ in cell.coords:
                mask[r - minr, c_ - minc] = True

            aligned_crop = np.zeros((h, w), dtype=crop.dtype)
            aligned_mask = np.zeros((h, w), dtype=bool)
            r_offset = minr - minr_total
            c_offset = minc - minc_total
            aligned_crop[r_offset:r_offset + crop.shape[0], c_offset:c_offset + crop.shape[1]] = crop
            aligned_mask[r_offset:r_offset + mask.shape[0], c_offset:c_offset + mask.shape[1]] = mask

            aligned_crops.append(aligned_crop)
            aligned_masks.append(aligned_mask)

        mask_intersection = np.logical_and.reduce(aligned_masks)

        summed_crop = np.sum(aligned_crops, axis=0)

        masked_crop = np.where(mask_intersection, summed_crop, 0)
        
        foci_count, foci_pos, foci_area = count_foci(masked_crop)
        foci_group_count = count_group_foci([f[0] for f in foci_pos])

        result_foci.append({
            "ID": fullcell.idcell,
            "Foci_Count": foci_count,
            "Foci_Group_Count": foci_group_count,
            "Foci_Area": foci_area,
            "Levels": len(fullcell.cells),
            "Area": np.count_nonzero(mask_intersection),
            "Zone": fullcell.region
        })
        
    return result_foci

def results_CSV(resultados_foci):
    df = pd.DataFrame(resultados_foci)
    df.to_csv("result_foci.csv", index=False)


def main():
    images_blue = load_img(dir_blue)
    images_red = load_img(dir_canal_c2)
    regions = seg_img(images_blue)
    conflicts = assign_conflicts(regions)
    fullcells = assign_cells(conflicts)
    foci_results = count_all_foci(fullcells, images_red)
    results_CSV(foci_results)

    
if __name__ == "__main__":
    main()