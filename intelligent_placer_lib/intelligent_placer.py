from intelligent_placer_lib.placer import *
def check_image(path):
    
    image = imread(path)
    box, rect = find_poly(image)
    # Если прямоугольник не нашелся - нам некуда складывать предметы, ответ алгоритма - False
    if rect is None:
        return False
   
    wf = filter_items(image, box)
    masks, areas = get_masks(wf)
    answer = placer(~rect, masks)
    return answer