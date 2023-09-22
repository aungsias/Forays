# A Comprehensive, Comparitive Study of Sentiment Analysis on Disney

Aung Si<br>
September 18<sup>th</sup>, 2023

---

## Introduction 

In an era where information is as volatile as it is abundant, the financial markets often find themselves at the mercy of public sentiment. News headlines, in particular, serve as a double-edged sword, possessing the power to both propel and plummet stock prices within moments. This study aims to explore this intricate relationship by focusing on Disney, a conglomerate whose diverse portfolio makes it a fascinating subject for this kind of analysis. The overarching questions guiding this study are: Which model is most apt at capturing headline sentiment, and can the sentiment derived from news headlines serve as a reliable indicator for Disney's stock performance?

## Objectives and Methods
The study unfolds in distinct yet interconnected phases, each with its own set of objectives. Initially, the focus is on determining the most apt sentiment analysis model. A collection of news headlines related to Disney serves as the dataset for this exploratory phase. Four machine learning models—TextBlob, BERT, Vader, and GPT-3—are employed for sentiment labeling. Their output is then qualitatively compared to my own evaluations to ascertain which model best captures the sentiment inherent in the headlines. 

Upon identifying the most efficacious model, the study shifts to data segregation based on emotional categories. The aim here is to excavate the lexicon most frequently associated with each sentiment, thereby providing a nuanced understanding of the language patterns that drive public opinion. Concurrently, the events corresponding to these dominant words are analyzed to offer context.

The final objective is to conduct a correlation analysis, linking the refined sentiment data to Disney's stock performance over a defined timeframe. This serves as the empirical test for the overarching hypothesis: Can news headline sentiment act as a reliable predictor for stock market movement?

## The Models
Four models have been selected for their varying degrees of sophistication and their widespread use in natural language processing tasks. Below is a delineation of each model and its specific relevance to sentiment analysis.

### 1. TextBlob
TextBlob operates on the simpler end of the spectrum, leveraging a rule-based approach. It quantifies sentiment into polarity and subjectivity metrics. Although rudimentary, its straightforward nature offers a baseline against which more complex models can be evaluated. TextBlob employs basic algorithms like Naive Bayes, which makes it computationally efficient, but this simplicity also limits its capability to grasp the intricacies of emotional tone. As a result, while it may lack the sophistication of more advanced models, TextBlob remains a valuable component in the study for its role in establishing a rudimentary benchmark.

### 2. BERT (Bidirectional Encoder Representations from Transformers)
BERT employs a deep learning architecture, using transformer layers to understand the context within which words appear. In contrast to TextBlob's rule-based approach, BERT's deep learning architecture equips it with the ability to discern complex patterns within the text. Its bidirectional nature allows for contextual analysis from both preceding and succeeding words in a sentence, thereby providing a more holistic view of sentiment. Given its computational intensity, BERT is often deployed for specialized tasks where a higher level of accuracy in sentiment classification is paramount.

### 3. Vader (Valence Aware Dictionary and sEntiment Reasoner)
Vader's unique strength lies in its ability to discern the emotional valence of colloquial language, slang, and internet jargon, providing a sentiment score that is often more in tune with human interpretation. Furthermore, it factors in text modifiers, such as capitalization and punctuation marks, to fine-tune its sentiment scores. Due to its specialized lexicon and rule set, Vader offers a more tailored approach for sentiment analysis in domains like social media or news headlines, where brevity and expressive language are common.

### 4. GPT-3 (Generative Pre-trained Transformer 3)
The most advanced model in our toolkit, GPT-3 is designed to perform a myriad of language tasks. In terms of sentiment analysis, its training on a vast corpus of text allows for a robust understanding of context and emotion. Its strength lies in its ability to adapt and generalize, providing a sophisticated lens through which headline sentiment can be examined. While not explicitly tailored for sentiment analysis, GPT-3's prowess stems from its extensive training on a colossal dataset. This enables it to have near-human understanding of context and emotion, despite its primary design for a broader array of language tasks. Its capacity to adapt and generalize renders it a potent tool for examining the sentiment in news headlines, albeit with the caveat that it is not specialized in this particular domain.

The juxtaposition of these models serves not only to identify the most effective tool for sentiment extraction but also to understand the trade-offs involved in selecting one model over another. The end goal is to furnish a rigorous methodology for capturing the fluidity that informs market dynamics.