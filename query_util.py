from SPARQLWrapper import SPARQLWrapper, JSON, POST
import json, os

sparql = SPARQLWrapper("http://172.17.0.10:8080/bigdata/sparql")                      
sparql.setReturnFormat(JSON)
sparql.setMethod(POST)
path_to_server = '/data/nginx/fileServer'


def process_query(queryString):
    sparql.setQuery(queryString);
    result = sparql.query().convert();
    var = result['head']['vars'][0]
    if 'image' in var:
        return process_image_query(result['results']['bindings'])
    else:
        return process_anno_query(result['results']['bindings'])


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

    for result in results:
        x = float(result['x']['value'])
        y = float(result['y']['value'])
        w = float(result['w']['value'])
        h = float(result['h']['value'])
        area = w * h
        bbox = [x, y, w, h]
        annId = len(annotations)
        imageId = int(imageURI.split('/')[-1])

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
        'id': len(categories),
        'name': label,
    }
    categories.append(category)
    return len(categories) - 1


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
    x = float(result['x']['value'])
    y = float(result['y']['value'])
    w = float(result['w']['value'])
    h = float(result['h']['value'])
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