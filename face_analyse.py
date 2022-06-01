from deepface import DeepFace
import cv2
import dlib
import copy
import time


def ltrb_to_xywh(pred):
	"""
	transform label left, top, right, bottom to x, y, width, height
	params:
		pred list(int): [left, top, right, bottom]
	return:
		pred list(int): [x, y, weight, height]
	"""
	return [pred[0], pred[1], pred[2] - pred[0], pred[3] - pred[1]]


def dlib_hog_pred(img, dlib_hog_model):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = dlib_hog_model(gray, 1)
	return faces


def face_analyse(img, db_path, draw=True, model_name="Facenet"):
	"""
	Params:
		img (cv2 img): image to analyse
		db_path (Path or db): database containing faces
		draw (Bool): draw positions / names on original image
		model_name (str): name of the model for face recognition, must be in 
		["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
	Return:
		img (cv2 img): original with/out drawing
		names (list str): list of names of person found in image
		crop_imgs (list cv2 img): list of image representing each person on the image
	"""
	dlib_hog_model = dlib.get_frontal_face_detector()
	faces = dlib_hog_pred(img, dlib_hog_model)
	color = (255, 0, 0)
	color_name = (0, 0, 255)
	crop_imgs = []
	names = []
	final_img = copy.deepcopy(img)
	for i, face in enumerate(faces):
		xywh = ltrb_to_xywh([face.left(), face.top(), face.right(), face.bottom()])
		nrange = 10
		min_percent = 0.1
		for percent in range(nrange):
			percent = max(percent / nrange, min_percent)
			w = int(xywh[2] * percent)
			h = int(xywh[3] * percent)
			crop_img = img[max(face.top()-h, 0):face.bottom()+h, max(face.left()-w, 0):face.right()+w]
			try:
				timer = time.time()
				df = DeepFace.find(img_path=crop_img, db_path=db_path, detector_backend="retinaface", model_name=model_name)
				print(f"DeepFace model {model_name} timer: {str(round(time.time() - timer, 4))} sec")
				if df.shape[0] > 0:
					name = df["identity"][0].split("/")[-1].split(".")[0].replace("_", " ")
				else:
					name = "Unknown"
				break
			except:
				name = "Unknown"
		crop_imgs.append(crop_img)
		name = str(i) + ". " + name
		names.append(name)
		if draw:
			final_img = cv2.rectangle(final_img, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
			final_img = cv2.putText(final_img, name, (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_name, 2)
	return final_img, names, crop_imgs
