import json
import os

def load_img_ann(ann_path='test/_annotations.coco.json'):
    """return [{img_name, [ (x, y, h, w, label), ... ]}]"""
    with open(ann_path) as fp:
        root = json.load(fp)
    img_dict = {}
    for img_info in root['images']:
        img_dict[img_info['id']] = {'name': img_info['file_name'], 'anns': []}
    for ann_info in root['annotations']:
        img_dict[ann_info['image_id']]['anns'].append(
            ann_info['bbox'] + [ann_info['category_id']])

    return img_dict


def show_img_ann(img_info,i):
    from PIL import Image
    import cv2
    print(img_info)

    with open('test/_annotations.coco.json') as fp:
        root = json.load(fp)
    categories = root['categories']
    category_dict = {int(c['id']): c['name'] for c in categories}

    img_path = os.path.join('test', img_info['name'])
    img = cv2.imread(img_path)
    for ann in img_info['anns']:
        x, y, w, h = ann[0:4]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y+h)
        print(x1,y1,x2,y2)
        cv2.rectangle(img, (x1,y1), (x2, y2), (0,0,255), 2)
    cv2.imwrite('visual/'+str(i)+'.jpg', img)


def main():
    img_dict = load_img_ann()
    keys = list(img_dict.keys())
    for i in range (len(keys)):
        show_img_ann(img_dict[keys[i]],i)


if __name__ == '__main__':
    main()