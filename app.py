from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
import firebase_admin
from firebase_admin import messaging
from firebase_admin import credentials, auth as firebase_auth, firestore
from tensorflow import keras
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from flask import request
from email.mime.base import MIMEBase
from email import encoders
import requests
import os
from dotenv import load_dotenv

load_dotenv()

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
modelv2 = None  # Global model variable
cred = credentials.Certificate(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
firebase_admin.initialize_app(cred)
db = firestore.client()
RECAPTCHA_SECRET_KEY = os.getenv("RECAPTCHA_SECRET_KEY")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {
  "origins": [
    "http://localhost:5173",
    "https://ecomist-rosy.vercel.app",
    "https://aerotech-ati.vercel.app",
  ]
}}, supports_credentials=True)

app.secret_key = os.getenv("FLASK_SECRET_KEY")

def load_model_once():
    global modelv2
    if modelv2 is None:
        modelv2 = keras.models.load_model("LettuceModelV2.keras")
        logger.info("✅ Model loaded (lazy load)")


# Define class labels and recommendations
CLASS_LABELS = ["Bacterial", "Fungal", "Healthy", "Cant Classified"]

RECOMMENDATIONS = {
    "Bacterial": {
        "type": "general",
        "diseases": [
            {
                "name": "Bacterial Leaf Spot",
                "description": "Small, water-soaked lesions that turn dark with yellow halos. Common in cool, wet conditions. Caused by Xanthomonas campestris pv. vitians.",
                "treatments": [
                    {
                        "label": "Organic",
                        "steps": [
                            "Remove affected leaves promptly.",
                            "Apply neem oil every 5–7 days.",
                            "Water at the base to avoid leaf wetness.",
                            "Increase spacing for better airflow."
                        ]
                    },
                    {
                        "label": "Chemical",
                        "steps": [
                            "Spray copper-based bactericide weekly.",
                            "Avoid overhead irrigation.",
                            "Sanitize tools regularly."
                        ]
                    }
                ]
            },
            {
                "name": "Bacterial Soft Rot",
                "description": "Wet, mushy rot with foul odor, especially at plant base. Caused by Pectobacterium carotovorum.",
                "treatments": [
                    {
                        "label": "Cultural",
                        "steps": [
                            "Remove and destroy infected plants.",
                            "Improve soil drainage.",
                            "Avoid overwatering.",
                            "Keep humidity low if possible."
                        ]
                    }
                ]
            },
            {
                "name": "Corky Root",
                "description": "Roots develop rough, corky lesions, stunting growth. Caused by Rhizomonas suberifaciens.",
                "treatments": [
                    {
                        "label": "Preventive",
                        "steps": [
                            "Use resistant lettuce cultivars.",
                            "Practice crop rotation.",
                            "Avoid planting in previously infected soil."
                        ]
                    }
                ]
            },
            {
                "name": "Marginal Leaf Blight",
                "description": "Water-soaked streaks along leaf edges leading to necrosis. Caused by Pseudomonas marginalis.",
                "treatments": [
                    {
                        "label": "General",
                        "steps": [
                            "Avoid overhead watering.",
                            "Harvest early when leaf damage is visible.",
                            "Apply bactericide if necessary."
                        ]
                    }
                ]
            },
            {
                "name": "Varnish Spot",
                "description": "Shiny dark lesions, especially under humid conditions. Caused by Pseudomonas cichorii.",
                "treatments": [
                    {
                        "label": "Cultural",
                        "steps": [
                            "Reduce leaf wetness periods.",
                            "Apply copper bactericide during humid spells.",
                            "Improve air movement within canopy."
                        ]
                    }
                ]
            }
        ]
    },
    "Fungal": {
        "type": "general",
        "diseases": [
            {
                "name": "Alternaria Leaf Spot",
                "description": "Dark lesions with concentric rings. Caused by Alternaria sonchi.",
                "treatments": [
                    {
                        "label": "Organic",
                        "steps": [
                            "Apply compost tea or neem oil.",
                            "Improve plant ventilation.",
                            "Remove affected leaves."
                        ]
                    },
                    {
                        "label": "Chemical",
                        "steps": [
                            "Spray with chlorothalonil or azoxystrobin fungicide weekly."
                        ]
                    }
                ]
            },
            {
                "name": "Anthracnose",
                "description": "Sunken leaf spots, especially on mature leaves. Caused by Microdochium panattonianum.",
                "treatments": [
                    {
                        "label": "Fungicide",
                        "steps": [
                            "Apply mancozeb or chlorothalonil.",
                            "Avoid overhead irrigation.",
                            "Rotate crops annually."
                        ]
                    }
                ]
            },
            {
                "name": "Bottom Rot",
                "description": "Brown rot at base of plant, often under wet soil. Caused by Rhizoctonia solani.",
                "treatments": [
                    {
                        "label": "Cultural",
                        "steps": [
                            "Improve soil drainage.",
                            "Avoid excessive nitrogen fertilization.",
                            "Space plants properly."
                        ]
                    }
                ]
            },
            {
                "name": "Cercospora Leaf Spot",
                "description": "Elongated spots with light centers and dark margins. Caused by Cercospora longissima.",
                "treatments": [
                    {
                        "label": "Organic",
                        "steps": [
                            "Spray garlic extract or compost tea.",
                            "Remove affected leaves.",
                            "Increase airflow between plants."
                        ]
                    }
                ]
            },
            {
                "name": "Damping-Off",
                "description": "Seedlings rot at soil level and collapse. Caused by Pythium spp. and Rhizoctonia solani.",
                "treatments": [
                    {
                        "label": "Preventive",
                        "steps": [
                            "Use sterile seed-starting mix.",
                            "Avoid overwatering.",
                            "Ensure good drainage."
                        ]
                    }
                ]
            },
            {
                "name": "Downy Mildew",
                "description": "Yellow spots on upper leaves, white fuzz underneath. Caused by Bremia lactucae.",
                "treatments": [
                    {
                        "label": "Organic",
                        "steps": [
                            "Use resistant varieties.",
                            "Apply compost tea or copper soap.",
                            "Avoid watering foliage."
                        ]
                    },
                    {
                        "label": "Chemical",
                        "steps": [
                            "Apply fosetyl-Al or mancozeb every 7–10 days."
                        ]
                    }
                ]
            },
            {
                "name": "Powdery Mildew",
                "description": "White, powdery growth on leaves. Caused by Erysiphe cichoracearum.",
                "treatments": [
                    {
                        "label": "Organic",
                        "steps": [
                            "Apply milk spray (1 part milk:9 parts water).",
                            "Use neem oil or potassium bicarbonate."
                        ]
                    }
                ]
            },
            {
                "name": "Gray Mold",
                "description": "Soft, brown decay covered in gray fuzz. Caused by Botrytis cinerea.",
                "treatments": [
                    {
                        "label": "Cultural",
                        "steps": [
                            "Remove decaying tissue promptly.",
                            "Improve airflow.",
                            "Avoid high humidity conditions."
                        ]
                    }
                ]
            },
            {
                "name": "Sclerotinia Drop",
                "description": "Wilting stem with white fungal growth and black sclerotia. Caused by Sclerotinia sclerotiorum.",
                "treatments": [
                    {
                        "label": "Fungicide",
                        "steps": [
                            "Apply boscalid or fluazinam.",
                            "Rotate with non-host crops.",
                            "Bury crop residue after harvest."
                        ]
                    }
                ]
            },
            {
                "name": "Septoria Leaf Spot",
                "description": "Small yellow spots with black fruiting bodies. Caused by Septoria lactucae.",
                "treatments": [
                    {
                        "label": "General",
                        "steps": [
                            "Apply chlorothalonil or mancozeb.",
                            "Remove infected leaves regularly.",
                            "Avoid overhead watering."
                        ]
                    }
                ]
            }
        ]
    },
    "Nutrient Deficiency": {
    "type": "general",
    "diseases": [
        {
            "name": "Nitrogen Deficiency",
            "description": "Leaves turn pale green to yellow starting from the older (lower) leaves. Stunted growth and thin stems.",
            "treatments": [
                {
                    "label": "Organic",
                    "steps": [
                        "Apply compost or well-rotted manure.",
                        "Use fish emulsion or blood meal.",
                        "Foliar feed with seaweed extract weekly."
                    ]
                },
                {
                    "label": "Synthetic",
                    "steps": [
                        "Apply a nitrogen-rich fertilizer like urea or ammonium nitrate.",
                        "Follow manufacturer instructions to avoid overfertilization."
                    ]
                }
            ]
        },
        {
            "name": "Phosphorus Deficiency",
            "description": "Leaves may appear dark green with purple or reddish discoloration. Poor root development and delayed maturity.",
            "treatments": [
                {
                    "label": "Organic",
                    "steps": [
                        "Add bone meal or rock phosphate to the soil.",
                        "Use composted manure.",
                        "Ensure soil pH is between 6.0–7.0 for phosphorus availability."
                    ]
                },
                {
                    "label": "Synthetic",
                    "steps": [
                        "Apply superphosphate or monoammonium phosphate.",
                        "Water in thoroughly to help uptake."
                    ]
                }
            ]
        },
        {
            "name": "Potassium Deficiency",
            "description": "Leaf edges turn yellow or brown (scorching), particularly in older leaves. Plants may be more prone to wilting and disease.",
            "treatments": [
                {
                    "label": "Organic",
                    "steps": [
                        "Incorporate wood ash or kelp meal into soil.",
                        "Use compost high in banana peels or leafy greens.",
                        "Apply liquid seaweed fertilizer weekly."
                    ]
                },
                {
                    "label": "Synthetic",
                    "steps": [
                        "Apply muriate of potash or potassium sulfate.",
                        "Avoid excess nitrogen which can worsen symptoms."
                    ]
                }
            ]
        },
        {
            "name": "Calcium Deficiency",
            "description": "Young leaves are distorted or crinkled. Tips of leaves may appear scorched. Lettuce heads may be soft and deformed.",
            "treatments": [
                {
                    "label": "Organic",
                    "steps": [
                        "Add crushed eggshells or ground limestone to the soil.",
                        "Apply foliar spray with calcium chloride or calcium nitrate.",
                        "Use compost with dairy waste if available."
                    ]
                }
            ]
        },
        {
            "name": "Magnesium Deficiency",
            "description": "Older leaves show yellowing between veins (interveinal chlorosis), often with a reddish tint.",
            "treatments": [
                {
                    "label": "Organic",
                    "steps": [
                        "Add Epsom salt (magnesium sulfate) to watering can (1 tbsp per gallon).",
                        "Topdress with compost containing dolomitic lime.",
                        "Foliar spray weekly until symptoms improve."
                    ]
                }
            ]
        },
        {
            "name": "Iron Deficiency",
            "description": "Newer (younger) leaves turn yellow between veins, but veins stay green. Growth may be stunted.",
            "treatments": [
                {
                    "label": "Organic",
                    "steps": [
                        "Apply iron chelates to soil or foliage.",
                        "Ensure pH is not too alkaline (ideal 6.0–6.5).",
                        "Use compost to improve soil microbial activity."
                    ]
                }
            ]
        }
    ]
},
    "Healthy": {
    "type": "status",
    "description": "The lettuce appears healthy with no visible signs of disease, deficiency, or stress.",
    "steps": [
        "Continue regular watering and feeding schedules.",
        "Maintain good air circulation and sunlight exposure.",
        "Inspect weekly for early signs of pests or diseases.",
        "Apply preventive foliar spray (e.g., compost tea or seaweed extract) once every 2 weeks."
    ]
},

"Not Lettuce": {
    "type": "status",
    "description": "The uploaded image is not recognized as a lettuce plant. It may be another crop or object.",
    "steps": [
        "Please ensure the image shows a clear top-down or front view of a lettuce crop.",
        "Avoid including human hands or background distractions.",
        "Try again with a new, well-lit photo."
    ]
},

"Mosaic Virus": {
    "type": "general",
    "description": "Mosaic virus causes mottled light and dark green patterns on leaves. Growth is stunted and leaves may curl.",
    "steps": [
        "Remove and destroy infected plants immediately.",
        "Control aphids and other insect vectors with neem oil or insecticidal soap.",
        "Avoid handling plants when wet to prevent spread.",
        "Use virus-resistant lettuce varieties in future plantings."
    ]
},

"Cannot Classify": {
    "type": "status",
    "description": "The system could not confidently classify the crop. The image may be unclear, or symptoms do not match known categories.",
    "steps": [
        "Ensure the plant is well-focused and well-lit in the image.",
        "Capture symptoms clearly (e.g., full leaf, spots, rot, etc.).",
        "Retake the photo from multiple angles if needed.",
        "If issues persist, consult an agricultural expert or extension worker."
    ]
}
}

@app.route("/")
def home():
    return "🌱 EcoMist Flask API is running!"


@app.route('/predict-v2', methods=['POST'])
def predict():
    
    try:
        load_model_once()  # ⬅️ Load only if not yet loaded
        if modelv2 is None:
            return jsonify({"error": "Model failed to load"}), 500
        
        # ✅ Authenticate Firebase user
        decoded_user = verify_token(request)
        uid = decoded_user['uid']
        email = decoded_user.get('email', 'Unknown')
        logger.info(f"🔐 Authorized user: {email} ({uid})")

        # 📷 Get the uploaded image
        file = request.files.get('file')
        if file is None:
            return jsonify({"error": "No file uploaded"}), 400

        image = Image.open(io.BytesIO(file.read())).convert("RGB").resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = modelv2.predict(img_array)

        # 🧠 Fallback logic using "Cant Classified" (Lettuce) class
        MIN_CONFIDENCE = 0.50
        LETTUCE_CLASS_INDEX = CLASS_LABELS.index("Cant Classified")

        predicted_index = np.argmax(predictions[0])
        predicted_label = CLASS_LABELS[predicted_index]
        confidence = float(predictions[0][predicted_index])
        lettuce_confidence = float(predictions[0][LETTUCE_CLASS_INDEX])

        fallback_used = False
        if confidence < MIN_CONFIDENCE and lettuce_confidence > 0.25:
            predicted_label = "Cant Classified"
            confidence = lettuce_confidence
            fallback_used = True
            logger.info("⚠️ Fallback to 'Cant Classified' due to low confidence")

        class_probabilities = {
            CLASS_LABELS[i]: float(predictions[0][i]) for i in range(len(CLASS_LABELS))
        }

        return jsonify({
            "status": "success",
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "class_probabilities": class_probabilities,
            "recommendations": RECOMMENDATIONS.get(predicted_label, {
    "type": "unknown",
    "description": "No recommendations found for this class.",
    "steps": []
}),
            "fallback_used": fallback_used  # 🟡 Optional, for frontend awareness
        })

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500


# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": modelv2 is not None
    })

@app.route('/model-info-v2', methods=['GET'])
def model_info():
    return jsonify({
        "model_name": "LettuceModelV2",
        "input_shape": [224, 224, 3],
        "classes": CLASS_LABELS,
        "version": "1.0.0"
    })
def verify_token(request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise ValueError("Missing or malformed Authorization header")

    id_token = auth_header.split(" ")[1]
    decoded_token = firebase_auth.verify_id_token(id_token)
    return decoded_token  # includes uid, email, etc.

@app.route('/send-notification', methods=['POST'])
def send_notification():
    try:
        decoded_user = verify_token(request)
        print(f"🔐 Verified sender: {decoded_user.get('email')}")

        data = request.json
        fcm_token = data.get('token')
        title = data.get('title')
        body = data.get('body')

        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=fcm_token,
        )

        response = messaging.send(message)
        return jsonify({'success': True, 'response': response}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
def send_breach_email(uid, to_email, sensor_name, device_name, value, threshold):
    try:
        # 🔍 Get sender account from Firestore
        doc_ref = db.collection("mail_senders").document(uid)
        doc_data = doc_ref.get().to_dict()

        if not doc_data or 'accounts' not in doc_data or 'defaultEmail' not in doc_data:
            raise ValueError("Sender credentials not configured")

        default_email = doc_data['defaultEmail']
        sender_account = next((acc for acc in doc_data['accounts'] if acc['email'] == default_email), None)

        if not sender_account:
            raise ValueError("Default sender not found")

        sender_email = sender_account['email']
        app_password = sender_account['appPassword']

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"⚠️ Breach Alert: {sensor_name} on Your Device"
        body = f"""
        Hello,

        A sensor on your device has exceeded the set threshold.

        🧪 Sensor: {sensor_name}
        📈 Current Value: {value}
        🚫 Threshold: {threshold}
        📟 Device: {device_name}
        🕒 Time: {timestamp}

        Please take immediate action to address this issue!
        Check your Eco-Mist dashboard for more details.

        Regards,  
        Eco-Mist Monitoring System
        """

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, to_email, msg.as_string())

        logger.info(f"📧 Breach email sent to {to_email} from {sender_email}")

    except Exception as e:
        logger.error(f"❌ Failed to send breach email: {e}")
        raise
    
@app.route('/breach-email', methods=['POST'])
def breach_email():
    try:
        decoded_user = verify_token(request)
        email = decoded_user.get("email")
        uid = decoded_user.get("uid")

        data = request.json
        device_id = data.get("device_id")
        device_name = data.get("device_name")
        sensor_name = data.get("sensor_name")
        value = data.get("value")
        threshold = data.get("threshold")

        if not all([email, device_name, sensor_name, value, threshold]):
            return jsonify({"error": "Missing required fields"}), 400

        send_breach_email(uid, email, sensor_name, device_name, value, threshold)
        return jsonify({"success": True}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/send-reply', methods=['POST'])
def send_reply():
    try:
        # 🔐 Verify Admin
        decoded_user = verify_token(request)
        uid = decoded_user["uid"]

        # 📤 Email fields
        to_email = request.form.get('to')
        subject = request.form.get('subject')
        body = request.form.get('body')
        file = request.files.get('file')

        # 🔍 Get sender credentials from Firestore
        doc_ref = db.collection("mail_senders").document(uid)
        doc_data = doc_ref.get().to_dict()

        if not doc_data or 'accounts' not in doc_data or 'defaultEmail' not in doc_data:
            return jsonify({'error': 'No sender credentials configured'}), 400

        default_email = doc_data['defaultEmail']
        sender_account = next((acc for acc in doc_data['accounts'] if acc['email'] == default_email), None)

        if not sender_account:
            return jsonify({'error': 'Default sender not found'}), 400

        sender_email = sender_account['email']
        app_password = sender_account['appPassword']

        # 📧 Build email message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if file:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{file.filename}"')
            msg.attach(part)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.send_message(msg)

        logger.info(f"📤 Email sent from {sender_email} to {to_email}")
        return jsonify({'message': 'Email sent successfully'})

    except Exception as e:
        logger.error(f"❌ Email sending failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/verify-recaptcha', methods=['POST'])
def verify_recaptcha():
    data = request.get_json()
    token = data.get('token')

    response = requests.post(
        'https://www.google.com/recaptcha/api/siteverify',
        data={
            'secret': RECAPTCHA_SECRET_KEY,
            'response': token
        }
    )
    result = response.json()
    return jsonify(result)

@app.route('/predict-v3', methods=['POST'])
def predict_alt_model():
    try:
        # ✅ Firebase Auth
        decoded_user = verify_token(request)
        uid = decoded_user['uid']
        email = decoded_user.get('email', 'Unknown')
        logger.info(f"🔐 Authorized user: {email} ({uid})")

        # ✅ Get uploaded file
        image_file = request.files.get('image') or request.files.get('file')
        if not image_file:
            return jsonify({"error": "No image uploaded"}), 400

        file = image_file.read()

        # ✅ Load model if not cached
        alt_model = getattr(app, 'alt_model', None)
        if alt_model is None:
            alt_model = tf.saved_model.load("LettuceModelV3")
            app.alt_model = alt_model
        infer = alt_model.signatures["serving_default"]

        # ✅ Preprocess
        image = Image.open(io.BytesIO(file)).convert("RGB").resize((224, 224))
        array = np.array(image) / 255.0
        input_data = np.expand_dims(array, axis=0).astype(np.float32)
        input_tensor = tf.convert_to_tensor(input_data)

        # ✅ Predict
        input_key = list(infer.structured_input_signature[1].keys())[0]
        result = infer(**{input_key: input_tensor})
        predictions = list(result.values())[0].numpy()[0]

        # 🧠 Logic
        ALL_CLASSES = ["Bacterial", "Fungal", "Healthy", "Lettuce", "Mosaic Virus", "Not Lettuce", "Nutrient Deficiency"]
        LETTUCE_CLASSES = {"Bacterial", "Fungal", "Healthy", "Mosaic Virus", "Nutrient Deficiency", "Lettuce"}
        NOT_LETTUCE_CLASS = "Not Lettuce"
        CONFIDENCE_THRESHOLD = 0.50

        pred_index = np.argmax(predictions)
        predicted_label = ALL_CLASSES[pred_index]
        confidence = float(predictions[pred_index])

        # 🔁 Normalize for Model V2-style response
        final_label = predicted_label
        final_confidence = confidence

        if confidence >= CONFIDENCE_THRESHOLD and predicted_label != NOT_LETTUCE_CLASS:
            lettuce_indexes = [ALL_CLASSES.index(cls) for cls in LETTUCE_CLASSES if cls != "Lettuce"]
            filtered_probs = [predictions[i] for i in lettuce_indexes]
            best_index = lettuce_indexes[np.argmax(filtered_probs)]
            final_label = ALL_CLASSES[best_index]
            final_confidence = float(predictions[best_index])

        normalized_result = {
            "status": "success",
            "prediction": final_label,
            "confidence": round(final_confidence, 4),
            "class_probabilities": {
                ALL_CLASSES[i]: float(predictions[i]) for i in range(len(ALL_CLASSES))
            },
            "recommendations": RECOMMENDATIONS.get(final_label, {
    "type": "unknown",
    "description": "No recommendations found for this class.",
    "steps": []
}),
            "fallback_used": confidence < CONFIDENCE_THRESHOLD,
            "raw_model": "v3",
            "message": "🚫 This is not a lettuce." if final_label == NOT_LETTUCE_CLASS else ""
        }

        return jsonify(normalized_result)

    except Exception as e:
        logger.error(f"❌ Alt model prediction error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500



@app.route('/model-info-v3', methods=['GET'])
def model_info_alt():
    try:
        # 🔁 Load alt model if not yet cached
        alt_model = getattr(app, 'alt_model', None)
        if alt_model is None:
            alt_model = tf.saved_model.load("LettuceModelV3")
            app.alt_model = alt_model

        # 🧠 Return model structure info
        ALL_CLASSES = ["Bacterial", "Fungal", "Healthy", "Lettuce", "Mosaic Virus", "Not Lettuce", "Nutrient Deficiency"]
        SUBCLASS_MAP = {
            "Bacterial": ["Bacterial_Leaf_Spot", "Bacterial_Soft_Rot", "Wilt_and_Leaf_Blight"],
            "Fungal": ["Anthracnose", "Bottom Rot", "Downy Mildew", "Powdery Mildew", "Septoria Blight"],
            "Healthy": [], "Lettuce": [], "Mosaic Virus": [], "Nutrient Deficiency": [],
            "Not Lettuce": ["Apple", "Banana", "Bean", "Beetroot", "Carrot", "Tomato", "Watermelon"]
        }

        return jsonify({
            "model_name": "LettuceModelV3",
            "type": "SavedModel",
            "input_shape": [224, 224, 3],
            "classes": ALL_CLASSES,
            "subclass_map": SUBCLASS_MAP,
            "version": "2.0.0"
        })

    except Exception as e:
        logger.error(f"❌ Failed to get alt model info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def unified_predict():
    model_choice = request.form.get('model', 'v2')

    if model_choice == 'v2':
        return predict()
    elif model_choice == 'v3':
        return predict_alt_model()
    else:
        return jsonify({"error": f"Invalid model choice: {model_choice}"}), 400

@app.route('/model-info', methods=['GET'])
def unified_model_info():
    model_choice = request.args.get('model', 'v2')

    if model_choice == 'v2':
        return model_info()
    elif model_choice == 'v3':
        return model_info_alt()
    else:
        return jsonify({"error": "Invalid model choice"}), 400
CLASS_SEVERITY = {
    "Healthy": 0,
    "Nutrient Deficiency": 1,
    "Fungal": 2,
    "Bacterial": 3,
    "Mosaic Virus": 4,
    "Cant Classified": 5,
    "Not Lettuce": 6
}
@app.route('/predict-compare', methods=['POST'])
def predict_compare():
    try:
        # ✅ Verify user
        decoded_user = verify_token(request)
        uid = decoded_user["uid"]
        email = decoded_user.get("email", "Unknown")

        # ✅ Get files and model
        file_before = request.files.get('file_before')
        file_after = request.files.get('file_after')
        model_choice = request.form.get("model", "v2")

        if not file_before or not file_after:
            return jsonify({"error": "Both images are required"}), 400

        # ✅ Choose prediction route
        def predict_image(file, model_choice):
            with app.test_request_context(
                '/predict',
                method='POST',
                data={'model': model_choice, 'file': (file, "image.jpg")},
                headers={"Authorization": f"Bearer {request.headers.get('Authorization', '').split(' ')[1]}"}
            ):
                if model_choice == "v2":
                    return predict().json
                elif model_choice == "v3":
                    return predict_alt_model().json

        # ✅ Predict both images
        before_result = predict_image(file_before, model_choice)
        after_result = predict_image(file_after, model_choice)

        # 🧠 Progress Analysis
        before_label = before_result["prediction"]
        after_label = after_result["prediction"]

        severity_map = {
            "Healthy": 0,
            "Nutrient Deficiency": 1,
            "Fungal": 2,
            "Bacterial": 3,
            "Mosaic Virus": 4,
            "Cant Classified": 5,
            "Not Lettuce": 6
        }

        before_score = severity_map.get(before_label, 10)
        after_score = severity_map.get(after_label, 10)

        if before_label == after_label:
            progress = "same"
        elif after_score < before_score:
            progress = "improved"
        elif after_score > before_score:
            progress = "worsened"
        else:
            progress = "changed"

        suggestion = {
            "improved": "✅ Disease condition improved. Continue current care practices.",
            "worsened": "⚠️ Condition worsened. Apply urgent treatment based on new diagnosis.",
            "changed": "🔁 Disease type changed. Consider reanalyzing care strategy.",
            "same": "ℹ️ No change detected. Continue monitoring closely."
        }[progress]

        return jsonify({
            "status": "success",
            "model": model_choice,
            "before": before_result,
            "after": after_result,
            "progress": progress,
            "message": suggestion,
            "recommendation": RECOMMENDATIONS.get(after_label, {
                "type": "unknown",
                "description": "No recommendation available.",
                "steps": []
            })
        })

    except Exception as e:
        logger.error(f"❌ Compare prediction error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port)
#CORS(app, origins=["http://localhost:5173"])  # or your production URL  
