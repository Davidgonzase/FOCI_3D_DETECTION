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

#Variables loaded from Config, assignment needed for correct 
dir_blue = config.dir_blue
dir_canal_c2 = config.dir_canal_c2

#Function for loading tiff images using tifffile based on a path, used to read DAPI and FISH images
def load_img(path):  
    temp=[]
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".tif"):
            img = tiff.imread(os.path.join(path, filename))
            temp.append((filename, img))
    return temp

#As much ROI as needed
#IMPORTANT, SET IN CONFIG !!!
region1cords = config.region1cords
region2cords = config.region2cords
region3cords = config.region3cords

pathregions = []

#IMPORTANT!! CREATE AS MUCH REGIONS AND APPEND AS ROI NEEDED IN CONFIG
region1 = Path(region1cords)
region2= Path(region2cords)
region3 = Path(region3cords)

pathregions.append(region1)
pathregions.append(region2)
pathregions.append(region3)

#Functions for FOCI count provided by Alex Bernadí
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

#Main class for cell compositicon which contains cell slices, id and ROI based on given
class Fullcell:
    _id_counter = 0
    def __init__(self,region):
        Fullcell._id_counter += 1
        self.idcell = Fullcell._id_counter
        self.cells = []
        self.region=region

#Main class for cell slice with the following attributes:
#Coords: Set of coordinates that form the geometric figure corresponding to the layer
#Centroid: Coordinates corresponding to the geometric center of the evaluated layer
#Area: Size in square pixels of the area occupied by the layer.
#Locallabel: Identifier created during segmentation, it is not unique in the stack since the classification process is restarted for each iteration of the segmentation program, so the same identifier can be in several levels at once
#Bbox: The bbox are the coordinates that form the rectangle where the layer is located in the image
#Id: Unique value for each independent cell layer, used to uniquely identify layers
#Level: Corresponding to the position in the stack where the image in which the evaluated layer is located is located
#Lastcandidates: List of candidates who are levels above, the local identifier of the region is stored
#Newcandidates: List of candidates who are levels below, the local identifier of the region is stored

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

#Fucntion for Watershed and OTSU segmentation

def seg_img(imgblue):
    #List to store all detected regions at the end of the segmentation
    allregions = []
    
    #For each image in the stack
    for name, img in imgblue:
        #Normalization, contrast enhancement, and grayscale switching
        img_norm = (img - img.min()) / (img.max() - img.min())
        enhanced = exposure.equalize_adapthist(img_norm)
        gray = (enhanced * 255).astype(np.uint8)
        
        #OTSU segmentation based from area study, if resolution and zoom changed another study will be needed to changed min_size so cleaning is done, number used rounding down the lowest number and applying 15% threshold
        thresh_value = threshold_otsu(gray)
        binary = gray > thresh_value
        binary = remove_small_objects(binary, min_size=68)
        
        #Local maxima based on mean radius and diameter from the previus study, same as before if images changed another study must be conducted assuming cells as circles
        distance = ndi.distance_transform_edt(binary)
        coords = peak_local_max(
        distance, 
        labels=binary, 
        footprint=np.ones((15, 15)), 
        min_distance=7)
        local_maxi = np.zeros_like(distance, dtype=bool)
        local_maxi[tuple(coords.T)] = True
        markers = ndi.label(local_maxi)[0]
        
        #Appliying the local maxima to watershed
        labels = watershed(-distance, markers, mask=binary)
        refined_labels = np.zeros_like(labels)
        
        #Exclusion area based in the previous area threshold to exlude errors in Watershed, as always based on the mean area, change if needed
        for region in regionprops(labels):
            if region.area > 68:
                refined_labels[labels == region.label] = region.label
                
        regions = regionprops(refined_labels)

        #Exlusion based on ROI coords previously provided
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
    
#Fucntion for conflict assignment

def assign_conflicts(regions):
    cellist = []
    #For evety localregion
    for level in range(len(regions)):
        currentlevel = regions[level]
        for candidate in currentlevel:
            #Create cell and obtain coords set for intersection comparations
            temp = Cell(candidate.coords,candidate.centroid,level,candidate.area,candidate.label,candidate.bbox)
            candidate_coords_set = set(map(tuple, candidate.coords))
            
            #In case first not compare to nothing above
            if level != 0:
                previous_level = regions[level - 1]
                #For every cell in the previous level
                for prev in previous_level:
                    #Extact coords and based on how many pixels intersect and the area calc the overlap 
                    prev_coords_set = set(map(tuple, prev.coords))
                    intersection = candidate_coords_set & prev_coords_set
                    overlap_ratio = len(intersection) / candidate.area
                    #Overlap ratio considered, is best to make a study based over all
                    if overlap_ratio >= 0.5:
                        #Add the conflict to cell
                        temp.addlast(prev.label)
                        
            #In case last not compare to nothing below
            if level < len(regions) - 1:
                next_level = regions[level + 1]
                #For every cell in the next level
                for nxt in next_level:
                    nxt_coords_set = set(map(tuple, nxt.coords))
                    intersection = candidate_coords_set & nxt_coords_set
                    overlap_ratio = len(intersection) / candidate.area
                    #Overlap ratio considered, is best to make a study based over all
                    if overlap_ratio >= 0.5:
                        # the conflict to cell
                        temp.addnew(nxt.label)

            cellist.append(temp)
    return cellist

#Fucntion for assigning cells

def assign_cells(cellist):
    #Final cells and cells that where read
    final_fullcells = []
    used_labels = set()
    
    #For every cell
    for cell in cellist:
        #Dont read if already
        if cell.locallabel in used_labels:
            continue
        
        #Create chain 
        forward_chain = [cell]
        valid_chain = True
        current = cell

        #If has candidates foward
        while current.newcandidates:
            #Discard imperfect
            if len(current.newcandidates) > 1:
                valid_chain = False
                break
            #Search for conflicts in next cell
            next_label = current.newcandidates[0]
            next_cell = next((c for c in cellist if c.locallabel == next_label and c.level == current.level + 1), None)
            #Dont add not broken chains or duplicated conflicts 
            if not next_cell or next_cell.locallabel in used_labels:
                valid_chain = False
                break
            #Discard if it has folowing conflicts on same level
            if len(next_cell.lastcandidates) > 1:
                valid_chain = False
                break
            forward_chain.append(next_cell)
            current = next_cell

        #Same but with following level
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

        #If valid create fullcell 
        if valid_chain:
            region=0
            y, x = cell.centroid
            cords = (x, y)
            #Assing region
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

#Fucntion for FISH crop extraction and cell FOCI count

def count_all_foci(final_fullcells,imagesred):
    result_foci = []
    for fullcell in final_fullcells:
        #Creating general bbox with the min and max levels of all slice
        minr_total = min(c.bbox[0] for c in fullcell.cells)
        minc_total = min(c.bbox[1] for c in fullcell.cells)
        maxr_total = max(c.bbox[2] for c in fullcell.cells)
        maxc_total = max(c.bbox[3] for c in fullcell.cells)
        h = maxr_total - minr_total
        w = maxc_total - minc_total

        #List for creating all the crops and masks
        aligned_crops = []
        aligned_masks = []

        #For all slices in the cell
        for cell in fullcell.cells:
            level = cell.level
            #Extract content from red FISH images
            img_red = imagesred[level][1]
            minr, minc, maxr, maxc = cell.bbox
            
            #Create crop and mask
            crop = img_red[minr:maxr, minc:maxc]
            mask = np.zeros_like(crop, dtype=bool)
            
            #In order to not let any other posible near cells a mask with the slice postiion will be placed
            for r, c_ in cell.coords:
                mask[r - minr, c_ - minc] = True

            #Creation of the local bbox
            aligned_crop = np.zeros((h, w), dtype=crop.dtype)
            aligned_mask = np.zeros((h, w), dtype=bool)
            r_offset = minr - minr_total
            c_offset = minc - minc_total
            #Place mask and crop with the content of the red FISH img
            aligned_crop[r_offset:r_offset + crop.shape[0], c_offset:c_offset + crop.shape[1]] = crop
            aligned_mask[r_offset:r_offset + mask.shape[0], c_offset:c_offset + mask.shape[1]] = mask

            aligned_crops.append(aligned_crop)
            aligned_masks.append(aligned_mask)
            
        #Compute mask intersections
        mask_intersection = np.logical_and.reduce(aligned_masks)
        #Sum all crops
        summed_crop = np.sum(aligned_crops, axis=0)
        #Place masks on top of crops
        masked_crop = np.where(mask_intersection, summed_crop, 0)
        
        #Apply FOCI count functions
        foci_count, foci_pos, foci_area = count_foci(masked_crop)
        foci_group_count = count_group_foci([f[0] for f in foci_pos])

        #FOCi results for CSV
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

#Fucntion to transform results into CSV

def results_CSV(foci_results):
    df = pd.DataFrame(foci_results)
    df.to_csv("result_foci.csv", index=False)

#Functionality based on methodology

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