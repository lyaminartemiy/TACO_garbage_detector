def show_images_with_detection(number_of_batch: int, ids: list):
    """
        
    """
    for n_image in range(len(ids)):
        image_filepath = f'batch_{number_of_batch}/{ids[n_image]}.jpg'
        pylab.rcParams['figure.figsize'] = (28,28)

        # Obtain Exif orientation tag code
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        # Find image id
        img_id = -1
        for img in imgs:
            if img['file_name'] == image_filepath:
                img_id = img['id']
                break

        if img_id == -1:
            print(f"{image_filepath}: Incorrect file")
        else:
            print(image_filepath)
            I = Image.open(DATASET_PATH + '/' + image_filepath)

            fig, ax = plt.subplots(1)
            plt.axis("off")
            plt.imshow(I)

            annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
            anns_sel = coco.loadAnns(annIds)

            for ann in anns_sel:
                color = colorsys.hsv_to_rgb(np.random.random(), 1, 1)
                for seg in ann['segmentation']:
                    poly = Polygon(np.array(seg).reshape((int(len(seg)/2)), 2))
                    p = PatchCollection([poly], facecolor=color, edgecolors=color,
                                        linewidth=0, alpha=0.4)
                    ax.add_collection(p)
                    p = PatchCollection([poly], facecolor='none',
                                        edgecolors=color, linewidth=2)
                    ax.add_collection(p)

                [x, y, w, z] = ann['bbox']
                print("bbox coordinates:", *[x, y, w, z])
                rect = Rectangle((x, y), w, z, linewidth=2, edgecolor=color,
                                 facecolor='none', alpha=0.7, linestyle='--')
                ax.add_patch(rect)

        plt.show();
