from cv2 import TM_CCOEFF_NORMED, THRESH_BINARY, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, COLOR_GRAY2BGR, imwrite, imread, groupRectangles, resize, getRotationMatrix2D, warpAffine, threshold, moments, findContours, cvtColor, minMaxLoc, circle, matchTemplate, line, FONT_HERSHEY_SIMPLEX, putText, LINE_AA, approxPolyDP, threshold, medianBlur, dilate, GaussianBlur, THRESH_OTSU, arcLength, drawContours
from os import mkdir, listdir, getcwd, rmdir, remove
from os.path import join, basename, exists
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import pi, sin, cos
from imutils import grab_contours
from numpy import where, array, zeros, uint8
from json import load
from re import split

template_method = TM_CCOEFF_NORMED

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

get_fname = lambda path: basename(path)
get_fname_short = lambda path: get_fname(path).split('.')[0]

dir_path = lambda name: join(getcwd(), name)
file_path = lambda dirpath, filename: join(dirpath, filename)

blueprints_path = dir_path('Plans')
entrances_path = dir_path('Entrance')
templates_path = dir_path('Furniture')
output_path = dir_path('Output')

blueprints_files = [file_path(blueprints_path, blueprint_name) for blueprint_name in sorted_alphanumeric(listdir(blueprints_path))]
blueprints_files[:2]

templates_files = [file_path(templates_path, template_name) for template_name in listdir(templates_path)]
templates_files[:2]

entrance_files = [file_path(entrances_path, entrance_name) for entrance_name in listdir(entrances_path)]
entrance_files

def read(fname):
    return imread(fname, 0)

def scale(img, scale):
    return resize(img, (0,0), fx=scale, fy=scale)

def rotate_scale(img, angle, scale_xy=1):
    height, width = img.shape[:2]
    
    image_center = (width/2, height/2)
    
    rotation_mat = getRotationMatrix2D(image_center, angle, scale_xy)
    
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_img = warpAffine(img, rotation_mat, (bound_w, bound_h))
    return rotated_img

def detect_center(img):
    thresh = threshold(img, 60, 255, THRESH_BINARY)[1]
    cnts = findContours(thresh.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    c = cnts[0]
    
    M = moments(c)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    
    return cX, cY

def get_entrance(blueprint_img, settings):
    angle_precision = settings['angle_precision']
    scale_max, scale_min = settings['scale_max'], settings['scale_min']
    scale_precision = settings['scale_precision']
    threshold = settings['threshold']
    
    scale_max_marg = scale_max+scale_precision*0.1
    
    best_location = []
    best_entrance_dict = {'fname': '', 'x_mid': 0, 'y_mid': 0, 'scale': 0, 'angle': 0, 'score': 0}
    for entrance_fname in entrance_files:
        current_entrance = read(entrance_fname)
        
        for angl in range(0, 360, angle_precision):
            cur_entr_rot = rotate_scale(current_entrance, angl)
            
            scl = scale_min
            while scl < scale_max_marg:
                cur_entr = scale(cur_entr_rot, scl)
                result = matchTemplate(blueprint_img, cur_entr, template_method)
                _, max_val, _, location = minMaxLoc(result)
        
                if max_val <= best_entrance_dict['score'] or max_val <= threshold:
                    scl += scale_precision
                    continue
                    
                best_entrance_dict['fname'] = entrance_fname
                best_entrance_dict['scale'] = scl
                best_entrance_dict['angle'] = angl 
                best_entrance_dict['score'] = max_val 
                best_location = location
                scl += scale_precision
                
    best_entrance = read(best_entrance_dict['fname'])
    best_entrance = rotate_scale(best_entrance, best_entrance_dict['angle'])
    best_entrance = scale(best_entrance, best_entrance_dict['scale'])
    
    cX, cY = detect_center(best_entrance)
    best_entrance_dict['x_mid'] = best_location[1] + cY
    best_entrance_dict['y_mid'] = best_location[0] + cX
    
    return best_entrance_dict

def get_furnitures(blueprint_img, settings):
    angle_precision = settings['angle_precision']
    scale_max, scale_min = settings['scale_max'], settings['scale_min']
    scale_precision = settings['scale_precision']
    threshold_val = settings['threshold']
    max_instances_detect = settings['max_instances_detect']
    
    scale_max_marg = scale_max+scale_precision*0.1

    blueprint_img = threshold(blueprint_img, 128, 255, THRESH_BINARY | THRESH_OTSU)[1]
    detected_furnitures = {}
    for furniture_fname in templates_files:
        furniture_img = threshold(read(furniture_fname), 128, 255, THRESH_BINARY | THRESH_OTSU)[1]
        detected_furnitures[furniture_fname] = {}

        furniture_rects_all = []
        for angl in range(0, 360, angle_precision):
            cur_fur_rot = rotate_scale(furniture_img, angl)
            detected_furnitures[furniture_fname][angl] = {'score':0, 'scale':0, 'rectangles':[]}

            scl = scale_min
            while scl < scale_max_marg:
                cur_fur = scale(cur_fur_rot, scl)
                result = matchTemplate(blueprint_img, cur_fur, template_method)
                _, max_val, _, _ = minMaxLoc(result)

                max_det_score = detected_furnitures[furniture_fname][angl]['score']
                if max_val <= max_det_score or max_val <= threshold_val:
                    scl += scale_precision
                    continue

                yloc, xloc = where(result >= threshold_val)
                if len(yloc) > max_instances_detect:
                    scl += scale_precision
                    continue

                rectangles = []
                for (x, y) in zip(xloc, yloc):
                    rectangles.append([int(x), int(y), int(cur_fur.shape[1]), int(cur_fur.shape[0])])
                    rectangles.append([int(x), int(y), int(cur_fur.shape[1]), int(cur_fur.shape[0])])
                rectangles, _ = groupRectangles(rectangles, 1, 0.2)

                good_rects = []
                
                for r1 in rectangles:
                    polygon1 = Polygon([[r1[0], r1[1]], [r1[0], r1[1]+r1[3]], [r1[0]+r1[2], r1[1]+r1[3]], [r1[0]+r1[2], r1[1]]])
                    
                    rects_all = furniture_rects_all + good_rects 
                    for r2 in rects_all:
                        polygon2 = Polygon([[r2[0], r2[1]], [r2[0], r2[1]+r2[3]], [r2[0]+r2[2], r2[1]+r2[3]], [r2[0]+r2[2], r2[1]]])
                        scr = 2*polygon1.intersection(polygon2).area / (polygon1.area+polygon2.area)
                        if scr > settings['max_overlap_perc']:
                            break
                    else:
                        good_rects.append(r1)

                detected_furnitures[furniture_fname][angl]['score'] = max_val
                detected_furnitures[furniture_fname][angl]['scale'] = scl
                detected_furnitures[furniture_fname][angl]['rectangles'] = good_rects

                scl += scale_precision
            furniture_rects_all.extend(detected_furnitures[furniture_fname][angl]['rectangles'])
            if len(detected_furnitures[furniture_fname][angl]['rectangles']) == 0:
                detected_furnitures[furniture_fname].pop(angl, None)
        if not detected_furnitures[furniture_fname]:
            detected_furnitures.pop(furniture_fname, None)
    return detected_furnitures

def image_processing(img, settings):
    blueprint_walled = img.copy()
    thr = threshold(blueprint_walled, 250, 255, THRESH_BINARY | THRESH_OTSU)[1]
    blurred = medianBlur(thr, settings['median_blur'])
    dilated = dilate(blurred, None, iterations=1)
    blur = GaussianBlur(dilated, (settings['gauss_blur'], settings['gauss_blur']), 0)
    thresh = threshold(blur, settings['threshold'], 255, THRESH_BINARY)[1]
    return thresh

def polygonArea(X, Y):
    area = 0.0
    n = len(X)
    j = n - 1
    for i in range(0,n):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i 
    return abs(area / 2.0)

def remove_vertices(poly, min_distance):
    to_remove = []
    temp = list(poly[1:])
    temp.append(poly[0])
    for i_pt, (pt1, pt2) in enumerate(zip(poly, temp)):
        pt1 = pt1[0]
        pt2 = pt2[0]
        if pt1[0] == pt2[0] and pt1[1] == pt2[1]:
            to_remove.append(i_pt)
            continue
        distance = ((pt1[0]- pt2[0]) ** 2 + (pt1[1]- pt2[1]) ** 2) ** 0.5
        if distance < min_distance:
            to_remove.append(i_pt)
    last_idx = len(poly) - 1
    
    if len(to_remove) == 0:
        return array(poly)
        
    if to_remove[0] == 0 and to_remove[-1] == last_idx:
        to_remove = to_remove[:-1]
    
    poly = list(poly)
    for tr in to_remove[::-1]:
        poly.pop(tr)
    return array(poly)

def poly_correction(poly, offset):
    any_done = False
    temp = list(poly[1:])
    temp.append(poly[0])
    for i_pt, (pt1, pt2) in enumerate(zip(poly, temp)):
        pt1 = pt1[0]
        pt2 = pt2[0]
        for i, (xy1, xy2) in enumerate(zip(pt1, pt2)):
            if abs(xy1 - xy2) <= offset:
                if xy1 == xy2:
                    continue
                any_done = True
                poly[i_pt][0][i] = xy2
    if any_done:
        return poly
    return None

def correct_triangularities(poly):
    acted = True
    poly2 = poly.tolist()
    last_critical = 0
    while acted:
        next1 = list(poly2[1:])
        next1.append(poly2[0])
        next2 = list(next1[1:])
        next2.append(next1[0])
        next3 = list(next2[1:])
        next3.append(next2[0])
        size = len(poly2)
        acted = False
        is_critical = False
        for i_poly, (pt0, pt1, pt2, pt3) in enumerate(zip(poly2, next1, next2, next3)):
            pt0, pt1, pt2, pt3 = pt0[0], pt1[0], pt2[0], pt3[0]
            is_hor_12 = (pt1[0] - pt2[0]) == 0
            is_ver_12 = (pt1[1] - pt2[1]) == 0

            if not (is_hor_12 or is_ver_12):
                new_pt_1 = [pt1[0], pt2[1]]
                
                if not ((new_pt_1[0] == pt0[0] and new_pt_1[1] == pt0[1]) or (new_pt_1[0] == pt3[0] and new_pt_1[1] == pt3[1])):
                    is_hor_n10 = (pt0[0] - new_pt_1[0]) == 0
                    is_ver_n10 = (pt0[1] - new_pt_1[1]) == 0
                    is_hor_n13 = (pt3[0] - new_pt_1[0]) == 0
                    is_ver_n13 = (pt3[1] - new_pt_1[1]) == 0

                    if is_hor_n10 or is_ver_n10 or is_hor_n13 or is_ver_n13:
                        acted = True
                        temp = poly2[:(i_poly+2)%size]
                        temp.append([new_pt_1])
                        temp += poly2[(i_poly+2)%size:]
                        poly2 = temp
                        break
                
                new_pt_2 = [pt2[0], pt1[1]]
                
                if not((new_pt_2[0] == pt0[0] and new_pt_2[1] == pt0[1]) or (new_pt_2[0] != pt3[0] and new_pt_2[1] != pt3[1])):
                    is_hor_n20 = (pt0[0] - new_pt_2[0]) == 0
                    is_ver_n20 = (pt0[1] - new_pt_2[1]) == 0
                    is_hor_n23 = (pt3[0] - new_pt_2[0]) == 0
                    is_ver_n23 = (pt3[1] - new_pt_2[1]) == 0

                    if is_hor_n20 or is_ver_n20 or is_hor_n23 or is_ver_n23:
                        acted = True
                        temp = poly2[:(i_poly+2)%size]
                        temp.append([new_pt_2])
                        temp += poly2[(i_poly+2)%size:]
                        poly2 = temp
                        break
                else:
                    is_critical = True
                    is_hor_02 = (pt0[0] - pt2[0]) == 0
                    is_ver_02 = (pt0[1] - pt2[1]) == 0
                    is_hor_13 = (pt1[0] - pt3[0]) == 0
                    is_ver_13 = (pt1[1] - pt3[1]) == 0
                    if is_hor_02 or is_ver_02:
                        last_critical = i_poly + 1
                    elif is_hor_13 or is_ver_13:
                        last_critical = i_poly + 2
                    continue
        else:
            if not acted and is_critical:
                acted = True
                print("Deleting troublesome vertex")
                poly2.pop(last_critical)
    return poly2

def correct_lines(poly):
    acted = True
    poly2 = poly
    while acted:
        if len(poly) == 0:
            return []
        acted = False
        next1 = list(poly2[1:])
        next1.append(poly2[0])
        next2 = list(next1[1:])
        next2.append(next1[0])
        size = len(poly2)
        for i_poly, (pt0, pt1, pt2) in enumerate(zip(poly2, next1, next2)):
            pt0, pt1, pt2 = pt0[0], pt1[0], pt2[0]
            is_hor_01 = (pt0[0] - pt1[0]) == 0
            is_hor_12 = (pt1[0] - pt2[0]) == 0

            is_ver_01 = (pt0[1] - pt1[1]) == 0
            is_ver_12 = (pt1[1] - pt2[1]) == 0
            
            if (is_hor_01 and is_hor_12) or (is_ver_01 and is_ver_12):
                acted = True
                poly2.pop((i_poly+1)%size)
                break
    return poly2

def draw_angled_line(img, pt0, pt1, color=(0,0,0), thickness=2):
    line(img, pt0, pt1, color, thickness)

def get_polygons(img, settings):
    debug_img1 = ~zeros(img.shape, dtype=uint8)
    debug_img2 = ~zeros(img.shape, dtype=uint8)
    debug_img3 = ~zeros(img.shape, dtype=uint8)
    
    poly_approx = settings['poly_approx']
    min_area = settings['min_area']
    max_area = img.shape[0] * img.shape[1] * settings['max_area_percent']
    px_near_delete = settings['remove_near_vertex']
    linearize_offset = settings['linearize_offset' ]
    
    selected_polygons = []
    contours,h = findContours(img,1,2)
    for cnt in contours:
        # Processing
        approx = approxPolyDP(cnt,poly_approx*arcLength(cnt,True), closed=True)
        vertices = array(approx)
        if len(approx) < 4:
            continue
        area = polygonArea(vertices[:, :, 0], vertices[:, :, 1])

        if area < min_area:
             continue
        if area > max_area:
            continue

        cleaned_poly = remove_vertices(approx.copy(), px_near_delete)

        if len(cleaned_poly) < 4:
            continue

        corrected = poly_correction(cleaned_poly, linearize_offset)
        if corrected is None:
            continue

        corrected_full = correct_triangularities(corrected)
        corrected_full = correct_lines(corrected_full)
        if len(corrected_full) == 0:
            continue
        selected_polygons.append(corrected_full)

        drawContours(debug_img1,[cnt],0,(0,0,255),2)
        
        temp = list(approx[1:])
        temp.append(approx[0])
        for pt0, pt1 in zip(approx, temp):
            draw_angled_line(debug_img2, pt0[0], pt1[0])
        
        temp = list(corrected_full[1:])
        temp.append(corrected_full[0])
        for pt0, pt1 in zip(corrected_full, temp):
            draw_angled_line(debug_img3, pt0[0], pt1[0])

        for xy in corrected_full:
            pass
            circle(debug_img3, (xy[0][0], xy[0][1]), 9, (0, 0, 255), -1)
    return selected_polygons, [debug_img1, debug_img2, debug_img3]

def construct_rectangles(poly):
    if len(poly)==4:
        return [[p[0] for p in poly]]
    rects = []
    acted = True
    
    while acted:
        poly = correct_lines(poly)
        size = len(poly)
        if size == 4:
            rects.append([p[0] for p in poly])
            return rects
        acted = False
        
        poly_vertices = [(p[0][0], p[0][1]) for p in poly]
        polygon = Polygon(poly_vertices)
        
        for idx, pt in enumerate(poly):
            pt0 = pt[0]
            pt1 = poly[(idx+1)%size][0]
            pt2 = poly[(idx+2)%size][0]
            pt3 = poly[(idx+3)%size][0]

            is_hor_01 = (pt0[0] - pt1[0]) == 0
            is_hor_12 = (pt1[0] - pt2[0]) == 0
            is_hor_23 = (pt2[0] - pt3[0]) == 0

            is_ver_01 = (pt0[1] - pt1[1]) == 0
            is_ver_12 = (pt1[1] - pt2[1]) == 0
            is_ver_23 = (pt2[1] - pt3[1]) == 0

            if is_hor_01 and is_ver_12 and is_hor_23:
                dis01 = pt0[1]-pt1[1]
                dis21 = pt3[1]-pt2[1]
                if dis01 * dis21 < 0:
                    continue
                
                dis01 = abs(dis01)
                dis21 = abs(dis21)
                    
                mean_x = (pt0[0] + pt3[0]) // 2
                
                choosing = dis01<dis21
                min_y = pt0[1] if choosing else pt3[1]
                
                point = Point(mean_x, min_y)
                if not polygon.covers(point):
                    continue
                
                if dis01 == dis21:
                    rects.append([pt0, pt1, pt2, pt3])
                    poly.remove([pt1])
                    poly.remove([pt2])
                    acted = True
                    break
                elif dis01 < dis21:
                    rects.append([pt0, pt1, pt2, [pt3[0], pt0[1]]])
                    poly.insert((idx+1), [[pt3[0], pt0[1]]])
                    poly.remove([pt1])
                    poly.remove([pt2])
                    acted = True
                    break
                else:
                    rects.append([[pt0[0], pt3[1]], pt1, pt2, pt3])
                    poly.insert((idx+1), [[pt0[0], pt3[1]]])
                    poly.remove([pt1])
                    poly.remove([pt2])
                    acted = True
                    break
                    
            elif is_ver_01 and is_hor_12 and is_ver_23:
                dis01 = pt0[0]-pt1[0]
                dis21 = pt3[0]-pt2[0]
                if dis01 * dis21 < 0:
                    continue
                
                dis01 = abs(dis01)
                dis21 = abs(dis21)
                    
                mean_y = (pt0[1] + pt3[1]) // 2
                
                choosing = dis01<dis21
                min_x = pt0[0] if choosing else pt3[0]
                
                point = Point(min_x, mean_y)
                
                if not polygon.covers(point):
                    continue
                if dis01 == dis21:
                    rects.append([pt0, pt1, pt2, pt3])
                    poly.remove([pt1])
                    poly.remove([pt2])
                    acted = True
                    break
                elif dis01 < dis21:
                    rects.append([pt0, pt1, pt2, [pt0[0], pt3[1]]])
                    poly.insert((idx+1), [[pt0[0], pt3[1]]])
                    poly.remove([pt1])
                    poly.remove([pt2])
                    acted = True
                    break
                else:
                    rects.append([[pt3[0], pt0[1]], pt1, pt2, pt3])
                    poly.insert((idx+1), [[pt3[0], pt0[1]]])
                    poly.remove([pt1])
                    poly.remove([pt2])
                    acted = True
                    break
    print("Critical error. Contact the owner of the program")
    return rects

def get_walls(img, settings):
    ready_img = image_processing(img, settings)
    polygons, debug_imgs = get_polygons(ready_img, settings)
    
    rects = []
    for p in polygons:
        rects.extend(construct_rectangles(p))

    debug_img4 = ~zeros((*img.shape, 3), dtype=uint8)
    rect_dicts_list = []
    for rect in rects:
        draw_angled_line(debug_img4, rect[0], rect[1])
        draw_angled_line(debug_img4, rect[1], rect[2])
        draw_angled_line(debug_img4, rect[2], rect[3])
        draw_angled_line(debug_img4, rect[3], rect[0])
        
        x_mid = (rect[0][0] + rect[2][0]) / 2
        y_mid = (rect[0][1] + rect[2][1]) / 2
        width = abs(rect[0][0] - rect[2][0])
        height = abs(rect[0][1] - rect[2][1])
        rect_dict = {'x_mid': x_mid, 'y_mid': y_mid, 'width': width, 'height': height}
        rect_dicts_list.append(rect_dict)
        circle(debug_img4, (int(x_mid), int(y_mid)), 9, (0, 0, 255), -1)
    
    debug_imgs.append(debug_img4)
    return rect_dicts_list, debug_imgs

def draw_angled_rec(img, x0, y0, width, height, color, thickness, angle):
    _angle = angle * pi / 180.0
    b = cos(_angle) * 0.5
    a = sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    line(img, pt0, pt1, color, thickness)
    line(img, pt1, pt2, color, thickness)
    line(img, pt2, pt3, color, thickness)
    line(img, pt3, pt0, color, thickness)

def draw_center_point(img, center):
    fur_id = list(center.keys())[0]
    x, y = int(center[fur_id][0]), int(center[fur_id][1])
    black = (0,0,0)
    font_color = black
    font = FONT_HERSHEY_SIMPLEX
    font_size = 1.1
    font_thickness = 2
    img_text = putText(img, str(fur_id), (x+5,y-5), font, font_size, font_color, font_thickness, LINE_AA)
    return circle(
        img_text, 
        (x, y), 
        radius=0, color=black, thickness=8
    )

def draw_entrance_point(img, entrance):
    return circle(
        img, 
        (int(entrance['y_mid']), int(entrance['x_mid'])), 
        radius=0, color=(0, 0, 0), thickness=8
    )

def write_debug_info(blueprint_path, blueprint_img, entrance, furnitures, centers, debug_imgs):
    blueprint_fname = get_fname_short(blueprint_path)
    debug_path = file_path(dir_path('Output'), f'{blueprint_fname}_debug')
    if not exists(debug_path):
        mkdir(debug_path)
    else:
        clear_stats(debug_path)
    
    current_blueprint_bgr = cvtColor(blueprint_img, COLOR_GRAY2BGR)
    full_debug_img = current_blueprint_bgr.copy()
    full_debug_fname = f'{blueprint_fname}_full.png'
    full_debug_path = file_path(debug_path, full_debug_fname)
    draw_entrance_point(full_debug_img, entrance)

    i=0
    for fur_path, fur_dict in furnitures.items():
        furniture_fname = get_fname_short(fur_path)

        furniture_debug_path = file_path(debug_path, furniture_fname)
        if not exists(furniture_debug_path):
            mkdir(furniture_debug_path)
        
        for angle, detect_dir in fur_dict.items():
            fur_debug_img = current_blueprint_bgr.copy()
            fur_debug_img = draw_entrance_point(fur_debug_img, entrance)
            fur_debug_fname = f'{furniture_fname}_{angle}.png'
            fur_debug_path = file_path(furniture_debug_path, fur_debug_fname)

            for r in detect_dir['rectangles']:
                draw_angled_rec(fur_debug_img, r[0]+r[2]//2, r[1]+r[3]//2, r[2], r[3], (0,0,255), 3, angle)
                draw_angled_rec(full_debug_img, r[0]+r[2]//2, r[1]+r[3]//2, r[2], r[3], (0,0,255), 3, angle)
                fur_debug_img = draw_center_point(fur_debug_img, centers[i])
                full_debug_img = draw_center_point(full_debug_img, centers[i])
                i += 1
            imwrite(fur_debug_path, fur_debug_img)
    imwrite(full_debug_path, full_debug_img)
    
    debug_img_names = ['contours', 'poly', 'rect_poly', 'rects']
    for img, din in zip(debug_imgs, debug_img_names):
        full_path = file_path(debug_path, f'{blueprint_fname}_{din}.png')
        imwrite(full_path, img)

def get_middle_rel_pt(rec_center, entrance_xy, px_to_meter):
    rec_rel_center = rec_center[0] - entrance_xy[0], rec_center[1] - entrance_xy[1]
    rec_rel_center = rec_rel_center[0] * px_to_meter, rec_rel_center[1] * px_to_meter
    return rec_rel_center

def save_detection(blueprint_path, blueprint_img, entrance, furnitures, rect_dicts_list, debug_imgs, px_to_meters):
    if not exists(dir_path('Output')):
        mkdir(dir_path('Output'))
        
    stat_fname = get_fname_short(blueprint_path) + '.txt'
    stat_path = file_path(dir_path('Output'), stat_fname)
    
    fur_id = 1
    
    entrance_xy = entrance['y_mid'], entrance['x_mid']
    
    centers = []
    with open(stat_path, 'w') as f:
        
        for fur_path, fur_dict in furnitures.items():
            furniture_fname = get_fname_short(fur_path)
            
            for angle, detect_dir in fur_dict.items():

                for r in detect_dir['rectangles']:
                    rec_center = get_center_pt(r)
                    fur_center = get_middle_rel_pt(rec_center, entrance_xy, px_to_meters)
                    centers.append({fur_id: rec_center})
                    f.write(f'{fur_id};{furniture_fname};{round(fur_center[0], 5)};{round(fur_center[1], 5)};{angle};{round(detect_dir["scale"], 5)};{round(detect_dir["score"],5)}\n')
                    fur_id += 1
        for rect_dict in rect_dicts_list:
            rec_center = get_middle_rel_pt((rect_dict["x_mid"], rect_dict["y_mid"]), entrance_xy, px_to_meters)
            wid, hei = rect_dict["width"]*px_to_meters, rect_dict["height"]*px_to_meters
            f.write(f'{fur_id};{"sciana"};{round(rec_center[0], 5)};{round(rec_center[1], 5)};{round(wid, 5)};{round(hei, 5)};\n')
            fur_id += 1
    write_debug_info(blueprint_path, blueprint_img, entrance, furnitures, centers, debug_imgs)

def get_center_pt(rec):
    return rec[0] + rec[2]/2, rec[1] + rec[3]/2

def clear_stats(stat_path):
    furniture_paths = [join(stat_path, p) for p in listdir(stat_path) if '.' not in p]
    for fp in furniture_paths:
        filelist = [f for f in listdir(fp)]
        for f in filelist:
            remove(join(fp, f))
        rmdir(fp)
    plan_path = [join(stat_path, p) for p in listdir(stat_path) if '.' in p]
    if len(plan_path)>0:
        remove(plan_path[0])

def get_settings():
    return load(open('settings.json'))

def main():
    settings = get_settings()
    print('Rozpoczęto program')
    for bf_path in blueprints_files:
        print()
        print(f'Rozpoczynam plan: {get_fname(bf_path)}')
        blueprint_path = bf_path
        current_blueprint = read(blueprint_path)
        print('Szukam punktu początkowego...')
        entrance = get_entrance(current_blueprint, settings['entrance_detection'])
        print('Znalazłem!')
        print(f'Najlepsza skala: {entrance["scale"]}')
        print('Szukam mebli...')
        furnitures = get_furnitures(current_blueprint, settings['furniture_detection'])
        print('Znalazłem!')
        print('Wykrywam ściany...')
        rect_dicts_list, debug_imgs = get_walls(current_blueprint, settings['wall_detection'])
        print('Wykryłem!')
        print('Zapisuję...')
        save_detection(blueprint_path, current_blueprint, entrance, furnitures, rect_dicts_list, debug_imgs, settings['px_to_meters'])
        print('Zapisałem!')
    print('Koniec!')
    
if __name__ == '__main__':
    main()