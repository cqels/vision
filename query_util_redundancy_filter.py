from SPARQLWrapper import SPARQLWrapper, JSON, POST
import json, os

sparql = SPARQLWrapper("http://172.17.0.10:8080/bigdata/sparql")                      
sparql.setReturnFormat(JSON)
sparql.setMethod(POST)
path_to_server = '/data/nginx/fileServer'


def redundancy_filter(redundent_results):
    """ Filter single-image occurs many times and #part of# single object with many bboxes scenarios. """

    filtered_reuslts = []
    for result in redundent_results:
        if result not in filtered_reuslts:
            filtered_reuslts.append(result)

    return filtered_reuslts


def process_query(queryString):
    sparql.setQuery(queryString)
    result = sparql.query().convert()
    var = result['head']['vars'][0]
    # filter single image occurs many time
    filter_result = redundancy_filter(result['results']['bindings'])
    if 'image' in var:
        return process_image_query(filter_result)
    else:
        return process_anno_query(filter_result)


def process_image_query(bindings):
    categories = []
    images = []
    annotations = []
    for binding in bindings:
        imageURI = binding['image']['value']
        process_one_image(imageURI, images, annotations, categories)
        process_one_image_annotations(imageURI, images, annotations, categories)

    return {'images': images,
            'annotations': annotations,
            'categories': categories}


def process_one_image(imageURI, images, annotations, categories):
    queryImage = open('query/image.sparql').read().replace('?image', '<' + imageURI + '>')
    sparql.setQuery(queryImage)
    result = sparql.query().convert()['results']['bindings'][0]
    id = int(imageURI.split('/')[-1])
    w = int(result['w']['value'])
    h = int(result['h']['value'])
    file_name = result['fileName']['value']
    path = result['path']['value']
    image = {
        'id': id,
        'file_name': file_name,
        'image_path': path_to_server + path,
        'width': w,
        'height': h
    }
    images.append(image)


def process_one_image_annotations(imageURI, images, annotations, categories):
    queryAnno = open('query/image_anno.sparql').read().replace('?image', '<' + imageURI + '>')
    sparql.setQuery(queryAnno)
    results = sparql.query().convert()['results']['bindings']

    filter_results = redundancy_filter(results)

    total_imgId_xy_wh = []
    for result in filter_results:
        # issues caused by type conversion, because of that, some boxes can not be filtered
        w = round(float(result['w']['value']), 5)
        h = round(float(result['h']['value']), 5)
        x = round(float(result['x']['value']) - w*.5, 5)
        y = round(float(result['y']['value']) - h*.5, 5)
        area = round(w * h, 5)
        bbox = [x, y, w, h]
        annId = len(annotations)
        imageId = int(imageURI.split('/')[-1])
        # filter single-object-many-bboxes scenario
        imgId_xy_wh = [x, y, w, h, imageId]
        if imgId_xy_wh not in total_imgId_xy_wh:
            total_imgId_xy_wh.append(imgId_xy_wh)
        else:
            continue
        label = result['label']['value']
        category_id = add_category(label, categories)

        annotation = {
            'id': annId,
            'image_id': imageId,
            'bbox': bbox,
            'category_id': category_id,
            'area': area,
            'iscrowd': 0,
        }

        annotations.append(annotation)


def add_category(label, categories):
    for category in categories:
        if category['name'] == label:
            return category['id']

    category = {
        'supercategory': 'Thing',
        'id': len(categories) + 1,
        'name': label,
    }
    categories.append(category)
    return len(categories)


def process_anno_query(bindings):
    categories = []
    images = []
    annotations = []
    for binding in bindings:
        annoURI = binding['annotation']['value']
        process_one_annotation(annoURI, images, annotations, categories)
    return {'images': images,
            'annotations': annotations,
            'categories': categories}


def process_one_annotation(annoURI, images, annotations, categories):
    queryAnno = open('query/anno.sparql').read().replace('?annotation', '<' + annoURI + '>')
    sparql.setQuery(queryAnno)
    result = sparql.query().convert()['results']['bindings'][0]
    imageURI = result['image']['value']
    w = round(float(result['w']['value']), 5)
    h = round(float(result['h']['value']), 5)
    x = round(float(result['x']['value']) - w*.5, 5)
    y = round(float(result['y']['value']) - h*.5, 5)
    area = w * h
    bbox = [x, y, w, h]
    annId = len(annotations)
    image_id = int(imageURI.split('/')[-1])

    label = result['label']['value']
    category_id = add_category(label, categories)

    annotation = {
        'id': annId,
        'image_id': image_id,
        'bbox': bbox,
        'category_id': category_id,
        'area': area,
        'iscrowd': 0,
    }
    annotations.append(annotation)

    for image in images:
        if image['id'] == image_id:
            return

    process_one_image(imageURI, images, annotations, categories)