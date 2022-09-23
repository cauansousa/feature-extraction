def height_estimate(distance, height):
    distance += 32
    camera_image_height = 1.48 * distance
    if height < 640:
        subject_height = 640 - height
        hf = 1.52 - (((subject_height*camera_image_height)/1280)/100)
    else:
        subject_height = height - 640
        hf = (((subject_height*camera_image_height)/1280)/100) + 1.52
    print("Você tem entre %.2fm e %.2fm de altura." % ((hf - 0.02), (hf + 0.02)))

distancia_laser = 150
topo_bbox = 540

height_estimate(distancia_laser, topo_bbox)