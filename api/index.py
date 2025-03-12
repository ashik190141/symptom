import re
import difflib
import itertools
from flask import Flask, request, jsonify
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Define your existing symptom extraction functions here...
symptoms_keywords = [
    "chest pain", "chest tightness", "sharp chest pain", "pressure in chest", "aching chest",
    "shortness of breath", "difficulty breathing", "breathing problems", "feeling breathless",
    "extreme tiredness", "lack of energy", "weakness", "lethargy", "palpitations",
    "rapid heartbeat", "irregular heartbeat", "pounding heart", "heart racing", "dizziness",
    "light-headedness", "faint", "vertigo", "nausea", "vomiting", "upset stomach", "queasy",
    "swelling", "edema", "swollen feet", "puffy ankles", "heartburn", "indigestion", "acid reflux",
    "cold sweats", "night sweats", "tiredness", "unexplained weight gain", "loss of appetite",
    "difficulty sleeping", "insomnia", "sleep apnea", "bluish skin", "pale skin", "loss of consciousness",
    "confusion", "difficulty concentrating", "fainting", "near fainting", "heart murmur", "cold extremities",
    "numbness in limbs", "chronic cough", "wheezing", "coughing up pink or white mucus", "swollen abdomen",
    "increased blood pressure", "chest discomfort", "pain radiating to the back", "pain radiating to the neck", "feeling of fullness after meals", "anxiety",
    "restlessness", "high blood sugar", "high cholesterol", "increased risk of clotting", "lightheadedness",
    "swollen veins in the neck", "decreased exercise tolerance", "heart attack", "stroke", "nausea and vomiting",
    "fatigue", "cold sweats", "anxiety", "shortness of breath", "blurred vision", "difficulty speaking", "weakness in limbs", "trouble walking", "weak pulse",
    "pounding heart", "feeling dizzy", "fainting", "sweating", "heavy breathing",
    "shortness of air", "nauseous", "feeling weak", "difficulty concentrating", "coughing up blood",
    "heart pain", "unusual heartbeats", "feeling cold", "slow heart rate", "sweaty palms", "tight chest",
    "restless", "chronic tiredness", "clammy skin", "chest tightness", "fever","back pain","heart palpitations", "headache", "migraine", "throbbing headache", "sinus pain", "jaw pain", "toothache",
    "earache", "ringing in ears", "sore throat", "hoarseness", "difficulty swallowing",
    "abdominal pain", "cramps", "bloating", "diarrhea", "constipation", "blood in stool",
    "black stool", "rectal bleeding", "urinary problems", "frequent urination", "painful urination",
    "blood in urine", "dark urine", "joint pain", "stiffness", "muscle pain", "muscle weakness",
    "tingling sensation", "burning sensation", "rash", "itching", "hives", "redness", "swelling in joints",
    "bruising", "bleeding gums", "nosebleeds", "dry mouth", "dry eyes", "vision changes", "double vision",
    "sensitivity to light", "hearing loss", "memory loss", "mood swings", "depression", "irritability",
    "panic attacks", "hallucinations", "seizures", "tremors", "unsteady gait", "balance problems",
    "loss of taste", "loss of smell", "sore muscles", "stiff neck", "swollen lymph nodes",
    "frequent infections", "low blood pressure", "high blood pressure", "irregular periods",
    "heavy menstrual bleeding", "missed periods", "pain during intercourse", "vaginal discharge",
    "penile discharge", "testicular pain", "breast pain", "lump in breast", "lump in neck",
    "lump in armpit", "lump in groin", "unintentional weight loss", "excessive thirst",
    "excessive hunger", "frequent infections", "slow healing wounds", "hair loss", "brittle nails",
    "dry skin", "yellowing of skin", "yellowing of eyes", "dark circles under eyes", "puffy eyes",
    "swollen face", "swollen hands", "swollen legs", "swollen joints", "cracked lips",
    "mouth ulcers", "white patches in mouth", "bleeding from nose", "bleeding from gums",
    "bleeding from rectum", "bleeding from vagina", "bleeding from penis", "bleeding from ears",
    "bleeding from eyes", "blood in vomit", "blood in sputum", "blood in saliva", "blood in semen",
    "blood in vaginal discharge", "blood in urine", "blood in stool", "black stool", "tarry stool",
    "clay-colored stool", "greasy stool", "foul-smelling stool", "foul-smelling urine",
    "foul-smelling breath", "foul-smelling sweat", "foul-smelling vaginal discharge",
    "foul-smelling penile discharge", "foul-smelling ear discharge", "foul-smelling nasal discharge",
    "foul-smelling eye discharge", "foul-smelling sputum", "foul-smelling saliva",
    "foul-smelling semen", "foul-smelling stool", "foul-smelling urine", "foul-smelling breath",
    "foul-smelling sweat", "foul-smelling vaginal discharge", "foul-smelling penile discharge",
    "foul-smelling ear discharge", "foul-smelling nasal discharge", "foul-smelling eye discharge",
    "foul-smelling sputum", "foul-smelling saliva", "foul-smelling semen","stress","blood pressure"
]

negation_words = [
    "no", "not", "haven't", "doesn't", "don't", "didn't", "do not", "have not", "does not","did not",
    "never", "doesn't seem", "does not seem", "doesn't feel", "does not feel", "no history", "not currently", "none", "not at all", "no longer",
    "hasn't been", "without", "nothing", "did not", "never felt", "not experiencing", "no signs of", "don't feel",
    "not having", "never had", "isn't", "wasn't", "won't", "wouldn't", "does not", "not yet", "not yet felt",
    "nothing at all", "neither", "not anymore", "not any", "not in the past", "not like before", "none at all",
    "not sure", "not detected", "doesn't exist", "doesn't apply", "not present", "no issues", "no trouble",
    "not observed", "no signs", "didn't notice", "is not happening", "is not felt", "no discomfort", "no pain",
    "no problem", "not seen", "no change", "hasn't experienced", "hasn't noticed", "no indication",
    "no symptoms", "no abnormalities", "no pain or discomfort", "doesn't bother me", "no abnormality",
    "didn't have", "not bothered", "no discomfort or pain", "not yet felt", "not experiencing any","cannot", "can't", "could not", "couldn't", "should not", "shouldn't", "would not", "wouldn't",
    "might not", "mightn't", "must not", "mustn't", "need not", "needn't", "ought not", "oughtn't",
    "rarely", "hardly", "scarcely", "barely", "seldom","almost never","no evidence of", "no sign of", "no indication of", "no report of", "no mention of", "no complaints of",
    "no record of", "no history of", "no evidence", "no sign", "no indication", "no report", "no mention",
    "no complaint", "no record", "no history"
]

temporal_markers = [
    "yesterday", "last week", "last month", "last year", "last night",
    "the day before yesterday", "a few days ago", "a while ago",
    "earlier this week", "earlier this month", "earlier today",
    "recently", "in the past", "before", "earlier", "previously",
    "before now", "back then", "in days gone by", "formerly", "not anymore",
    "just now", "recent", "past few days", "for a few days", "for weeks",
    "for months", "for years", "over the past week", "over the past month",
    "since last week", "since yesterday", "since earlier", "prior to", "ago",
    "I had", "I used to", "I have had", "I experienced", "it went away",
    "it subsided", "I no longer feel", "I was diagnosed with",
    "after", "when I was younger", "during my", "prior to",
    "following", "at the time of","a week ago", "a month ago", "a year ago", "a couple of days ago",
    "a couple of weeks ago", "a couple of months ago", "a couple of years ago",
    "a long time ago", "a short while ago", "a moment ago", "a minute ago",
    "an hour ago", "a day ago", "a few hours ago", "a few minutes ago",
    "a few weeks ago", "a few months ago", "a few years ago", "a decade ago",
    "in the last week", "in the last month", "in the last year", "in the last decade",
    "in recent weeks", "in recent months", "in recent years", "in recent days",
    "in the previous week", "in the previous month", "in the previous year",
    "in the previous decade", "in the earlier part of", "in the later part of",
    "in the beginning of", "in the middle of", "in the end of", "in the early stages",
    "in the late stages", "in the initial phase", "in the final phase",
    "in the early days", "in the later days", "in the early years", "in the later years",
    "in the early months", "in the later months", "in the early weeks", "in the later weeks",
    "in the early hours", "in the later hours", "in the early minutes", "in the later minutes",
    "in the early seconds", "in the later seconds", "in the early moments", "in the later moments",
    "in the early stages of", "in the later stages of", "in the early phases of", "in the later phases of",
    "in the early periods of", "in the later periods of", "in the early times of", "in the later times of",
    "in the early days of", "in the later days of", "in the early weeks of", "in the later weeks of",
    "in the early months of", "in the later months of", "in the early years of", "in the later years of",
    "in the early decades of", "in the later decades of", "in the early centuries of", "in the later centuries of"
]

special_phrases = {
    "nothing but": False,
    "only": False,
    "just": False,
    "merely": False,
    "solely": False,
    "no other than": False,
    "except for": True,
    "other than": True,
    "besides": True,
    "apart from": True,
    "without": True,
    "not counting": True,
    "leaving out": True,
    "excluding": True,
    "barring": True,
    "save for": True,
    "other": True,
    "besides that": True,
    "apart from that": True,
    "in the absence of": True,
    "with the exception of": True,
    "all but": False,
    "little more than": False,
    "hardly any": True,
    "scarcely any": True,
    "barely any": True,
    "almost no": True,
    "virtually no": True,
    "practically no": True,
    "nearly no": True,
    "close to no": True,
    "next to no": True,
}

def combine_words_sentence(words_in_sentences):
    combined_words = []
    combined_words.extend([word for word in words_in_sentences if len(word) >= 3])

    for r in range(2, len(words_in_sentences) + 1):
        word_combinations = [' '.join(comb) for comb in itertools.combinations(words_in_sentences, r)]

        combined_words.extend([
            ' '.join([word for word in comb.split() if len(word) >= 3])
            for comb in word_combinations
            if len(comb.replace(' ', '')) >= 3 and
               len(comb.split()) <= 5 and
               len(comb.split()) >= 2
        ])

        combined_words = list(set(combined_words))
        combined_words = [word for word in combined_words if word.strip()]

    return combined_words

def handle_negation(combo_words, symptom, sentence):
    target = None
    for i in range(len(combo_words)):
        match_ratio = difflib.SequenceMatcher(None, combo_words[i], symptom).ratio() * 100
        if match_ratio >= 80 and combo_words[i] in sentence:
            target = combo_words[i]
            break

    if target:
        split_sentence = sentence.split(target)[0]
        words_before_target = split_sentence.split()
        last_five_words = words_before_target[-5:]
        last_five_sentence = ' '.join(last_five_words)

        for phrase, is_negation in special_phrases.items():
            if phrase in last_five_sentence.lower():
                if not is_negation:
                    return False

        for negation in negation_words:
            if negation in last_five_sentence:
                return True

        return False
    else:
        words_in_sentence = sentence.lower().split()
        first_five_words_with_join = ' '.join(words_in_sentence[:5])
        last_five_words_with_join = ' '.join(words_in_sentence[-5:])

        for negation in negation_words:
            if negation in first_five_words_with_join or negation in last_five_words_with_join:
                return True

        return False

def handle_temporal(combo_words, symptom, sentence):
    target = None
    for i in range(len(combo_words)):
        match_ratio = difflib.SequenceMatcher(None, combo_words[i], symptom).ratio() * 100
        if match_ratio >= 90 and combo_words[i] in sentence:
            target = combo_words[i]
            break

    if target:
        split_sentence = sentence.split(target)[0]
        words_before_target = split_sentence.split()
        last_five_sentence = ' '.join(words_before_target[-5:])

        split_sentence_after_target = sentence.split(target)[1]
        words_after_target = split_sentence_after_target.split()
        first_five_sentence = ' '.join(words_after_target[:5])

        for temporal in temporal_markers:
            if temporal in last_five_sentence or temporal in first_five_sentence:
                return True

        return False
    else:
        words_in_sentence = sentence.lower().split()
        first_five_words_with_join = ' '.join(words_in_sentence[:5])
        last_five_words_with_join = ' '.join(words_in_sentence[-5:])

        for temporal in temporal_markers:
            if temporal in first_five_words_with_join or temporal in last_five_words_with_join:
                return True

        return False


def extract_current_symptoms(conversation):
    symptoms_present = set()
    symptoms_past = set()
    symptoms_mentioned = set()
    symptoms_negation = set()
    matched_symptoms = {}
    vectorizer = TfidfVectorizer()

    cleaned_conversation = re.sub(r'(doctor:|patient:|\n)', '', conversation)
    sentences = re.findall(r'[^.!?]*[.!?]', cleaned_conversation)

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        words_in_sentence = re.findall(r'\b\w+\b', sentence)
        combo_words = combine_words_sentence(words_in_sentence)

        for symptom in symptoms_keywords:
            for combo_word in combo_words:
                match_ratio = difflib.SequenceMatcher(None, combo_word, symptom).ratio() * 100
                if match_ratio >= 60:
                    tfidf_matrix = vectorizer.fit_transform([combo_word, symptom])
                    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                    cosine_match_ratio = cosine_sim[0][0] * 100

                    if cosine_match_ratio >= 80 or match_ratio >= 90:
                        symptoms_mentioned.add(symptom)
                        isTrue = False
                        if '?' in sentence:
                            isTrue = True
                            sentence = sentences[i+1].strip()
                            words_in_sentence = re.findall(r'\b\w+\b', sentence)
                            combo_words = combine_words_sentence(words_in_sentence)

                        close_match = get_close_matches(symptom, symptoms_keywords, n=1, cutoff=0.8)[0]
                        if isTrue:
                            if handle_negation(combo_words, close_match, sentence):
                                symptoms_negation.add(close_match)
                            if handle_temporal(combo_words, close_match, sentence):
                                symptoms_past.add(close_match)
                            else:
                                symptoms_present.add(symptom)
                        elif handle_negation(combo_words, close_match, sentence):
                            symptoms_negation.add(close_match)
                        elif handle_temporal(combo_words, close_match, sentence):
                            symptoms_past.add(close_match)
                        else:
                            symptoms_present.add(symptom)

    return {
        "symptoms_present": list(symptoms_present),
        "symptoms_past": list(symptoms_past),
        "symptoms_negation": list(symptoms_negation),
        "symptoms_mentioned": list(symptoms_mentioned)
    }

@app.route('/extract-symptoms', methods=['POST'])
def extract_symptoms_api():
    try:
        data = request.get_json()
        if 'conversation' not in data:
            return jsonify({"error": "No conversation provided"}), 400

        conversation = data['conversation'].lower()
        result = extract_current_symptoms(conversation)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
