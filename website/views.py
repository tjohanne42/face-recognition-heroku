from flask import Blueprint, render_template, request, flash, make_response, Response
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import base64
from face_analyse import face_analyse
from .models import User, Image
from . import db
import sqlalchemy
import os
import re
import time
from tqdm import tqdm
from datetime import datetime


views = Blueprint('views', __name__)

@views.route('/')
def home():
	return render_template("home.html")


"""
face-recognition
"""


def cv2_to_str(cv2_img):
	is_success, img_buf_arr = cv2.imencode(".jpg", cv2_img)
	if not is_success:
		return None

	byte_img = img_buf_arr.tobytes()
	byte_img = base64.b64encode(byte_img)
	byte_img = byte_img.decode("utf-8")
	return byte_img


@views.route("/face-recognition", methods=["GET", "POST"])
def face_recognition():
	if request.method == "POST":
		# request for file
		pic = request.files['pic']
		if not pic:
			flash("Image not valid.", category="error")
			return render_template("face_recognition.html", page=1)

		# check file type
		filename = secure_filename(pic.filename)
		mimetype = pic.mimetype
		if not filename or not mimetype or (mimetype != "image/jpeg" and mimetype != "image/png"):
			flash("Type image not valid.", category="error")
			return render_template("face_recognition.html", page=1)

		# str to cv2
		img = pic.read()
		# img = np.fromstring(img, np.uint8)
		img = np.frombuffer(img, np.uint8)
		img = cv2.imdecode(img, 1)

		# resize
		scale_percent = 50
		width = int(img.shape[1] * scale_percent / 100)
		height = int(img.shape[0] * scale_percent / 100)
		dim = (width, height)
		img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

		# face recognition
		img, names, crop_imgs = face_analyse(img, "known", draw=True)

		# convert face_recognition ouput from cv2 to str
		img = cv2_to_str(img)
		if img is None:
			flash("Conversion failed.", category="error")
			return render_template("face_recognition.html", page=1)
		for i in range(len(crop_imgs)):
			crop_imgs[i] = cv2_to_str(crop_imgs[i])
			if crop_imgs[i] is None:
				flash("Conversion failed.", category="error")
				return render_template("face_recognition.html", page=1)
		return render_template("face_recognition.html", page=2, drawn_img=img, crop_img=crop_imgs, names=names)
	else:
		return render_template("face_recognition.html", page=1)
