# Spotify Emotions Analyzer

## Overview
Spotify Emotions Analyzer is a research project aimed at analyzing the emotional content of song lyrics from a user's Spotify listening history. Leveraging natural language processing (NLP) techniques and machine learning models, the project provides personalized emotional analysis for music enthusiasts.

## Authors
- Derek Costello
- Alexander Elguezabal
- Mark Case

## Methodology
The core methodology of the project revolves around using a BERT (Bidirectional Encoder Representations from Transformers) Large Language Model (LLM) trained on the Google GoEmotions dataset. This dataset contains human-annotated Reddit comments categorized into various emotional categories, serving as the basis for predicting emotional weights for song lyrics.

## Functionality
- Collects lyrical data directly from the user's Spotify listening history.
- Predicts emotional weights for sentences extracted from song lyrics using the BERT LLM.
- Aggregates emotional weight predictions to classify sentences into emotional categories.
- Presents emotional analysis results through interactive infographics within a React-based web application.
- Cloud-hosted web application and server for enhanced accessibility and user experience.

## Reference
For more information and detailed documentation, please refer to the project repository: [Spotify Emotions Analyzer Repository](https://github.com/Frostfire25/Spotify_NLP_Service)

## License
This project is licensed under the [MIT License](LICENSE).
