import streamlit as st
from transformers import pipeline


# ------------------------------
# Load Whisper Model
# ------------------------------
def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0)


# ------------------------------
# Load NER Model
# ------------------------------
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """


    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")


# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
    Returns:
        str: Transcribed text from the audio file.
    """
    whisper_model = load_whisper_model()  # Load the whisper model
    audio = uploaded_file.read()  # Read the uploaded file
    transcription = whisper_model(audio, return_timestamps = True) # get transcript
    return transcription['text']


# ------------------------------
# Entity Extraction
# ------------------------------
def extract_entities(text, ner_pipeline):
    """
    Extract entities from transcribed text using the NER model.
    Args:
        text (str): Transcribed text.
        ner_pipeline: NER pipeline loaded from Hugging Face.
    Returns:
        dict: Grouped entities (ORGs, LOCs, PERs).
    """
    entities = ner_pipeline(text) # process the text with the ner model
    grouped_entities = {"ORGs": [], "LOCs": [], "PERs": []} # create a dictionary for entities

    for entity in entities: # group entities by their type
        if entity['entity_group'] == 'ORG':
            grouped_entities['ORGs'].append(entity['word'])
        elif entity['entity_group'] == 'LOC':
            grouped_entities['LOCs'].append(entity['word'])
        elif entity['entity_group'] == 'PER':
            grouped_entities['PERs'].append(entity['word'])

    # Remove duplicates
    for word in grouped_entities:
        grouped_entities[word] = list(set(grouped_entities[word]))

    return grouped_entities


# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Meeting Transcription and Entity Extraction") # the title of the app


    STUDENT_NAME = "Yusuf Öksüzer"
    STUDENT_ID = "150220334"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**") # designing
    st.write("Upload a business meeting audio file to:")
    st.write("1. Transcribe the meeting audio into text.")
    st.write("2. Extract key entities such as Persons, Organizations, Dates and Locations.")


    file = st.file_uploader("Upload a audio file (WAV format)", type=["wav"]) # upload the wav file

    if file is not None:
        with st.spinner("Transcribing the audio file... This may take a minute"):  # Show spinner while processing
            whisper_model = load_whisper_model() # load models
            ner_model = load_ner_model()
            transcript = transcribe_audio(file)

        st.subheader("Transcription:")
        st.write(transcript) # print the transcript

        with st.spinner("Extracting entities..."):
            entities = extract_entities(transcript, ner_model) # extract entities

        st.subheader("Extracted Entities:")  # print all entities and print suitable output for non-existent
        if entities['ORGs']:
            st.write("**Organizations (ORGs):**")
            for org in entities['ORGs']:
                st.write(f"- {org}")
        else:
            st.write("No organizations found.")
        if entities['LOCs']:
            st.write("**Locations (LOCs):**")
            for loc in entities['LOCs']:
                st.write(f"- {loc}")
        else:
            st.write("No locations found.")
        if entities['PERs']:
            st.write("**Persons (PERs):**")
            for per in entities['PERs']:
                st.write(f"- {per}")
        else:
            st.write("No persons found.")


if __name__ == "__main__":
    main()
