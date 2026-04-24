 Home | CS 288 [Skip to main content](#main-content "#main-content") Link Menu Expand (external link) Copy Copied 

[CS 288](/sp26/ "/sp26/")

* [Home](/sp26/ "/sp26/")* [Assignments](/sp26/assignments/ "/sp26/assignments/")* [Project](/sp26/project/ "/sp26/project/")* [Course Info](/sp26/course_info/ "/sp26/course_info/")* [Staff](/sp26/staff/ "/sp26/staff/")
 This site uses [Just the Docs](https://github.com/just-the-docs/just-the-docs "https://github.com/just-the-docs/just-the-docs"), a documentation theme for Jekyll.

# Advanced Natural Language Processing

Spring 2026

**Instructor**: [Sewon Min](https://www.sewonmin.com/ "https://www.sewonmin.com/"), [Alane Suhr](https://www.alanesuhr.com/ "https://www.alanesuhr.com/") 
 **Class hours**: TuThu 15:30–17:00 (15:40–17:00 considering Berkeley time) 
 **Class location**: SODA 306 
 **Instructor OH**: Right after the lectures at SODA 306 
 **GSI OH**: Monday (12:30 - 1 PM), Wednesday (11:30 AM - 12 PM) | [Zoom Link](https://berkeley.zoom.us/j/98315614287 "https://berkeley.zoom.us/j/98315614287")

**Ed link**: [https://edstem.org/us/join/XvztdK](https://edstem.org/us/join/XvztdK "https://edstem.org/us/join/XvztdK") (Please use Ed for any class related questions) 
 **Gradescope link**: [Gradescope](https://www.gradescope.com/courses/1233780 "https://www.gradescope.com/courses/1233780") (Use code: J4N7E2)

**Lecture recordings**: [https://www.youtube.com/playlist?list=PLnocShPlK-Fv9YZIX7qdOyc2GJqnT3D-8](https://www.youtube.com/playlist?list=PLnocShPlK-Fv9YZIX7qdOyc2GJqnT3D-8 "https://www.youtube.com/playlist?list=PLnocShPlK-Fv9YZIX7qdOyc2GJqnT3D-8") (Needs Berkeley log in, lecture 1 coming soon)

**Final project**: Final project logistics and reference topics: [https://docs.google.com/document/d/1C8Dl6DX0_F5g3HDR-Gwr1fTmKGgscxzbU9AiUpvxV0k/edit?usp=sharing](https://docs.google.com/document/d/1C8Dl6DX0_F5g3HDR-Gwr1fTmKGgscxzbU9AiUpvxV0k/edit?usp=sharing "https://docs.google.com/document/d/1C8Dl6DX0_F5g3HDR-Gwr1fTmKGgscxzbU9AiUpvxV0k/edit?usp=sharing")

---

This course provides a graduate-level introduction to Natural Language Processing (NLP), covering techniques from foundational methods to modern approaches. We begin with core concepts such as word representations and neural network–based NLP models, including recurrent networks and attention mechanisms. We then study modern Transformer-based models, focusing on pre-training, fine-tuning, prompting, scaling laws, and post-training. The course concludes with recent advances in NLP, including retrieval-augmented models, reasoning models, and multimodal systems involving vision and speech.

**Prerequisites**: CS 288 assumes prior experience in machine learning and proficiency in PyTorch. Students should be familiar with neural networks, PyTorch, and NumPy; no introductory tutorials will be provided.

## Schedule (Tentative)

All deadlines are at 5:59 PM PST.

01/20 Tue: Introduction & n-gram LM: **[01_Intro](assets/slides/CS288_sp26_01_Intro.pdf "assets/slides/CS288_sp26_01_Intro.pdf")** **[02_ngram_LM](assets/slides/CS288_sp26_02_ngram_LM.pdf "assets/slides/CS288_sp26_02_ngram_LM.pdf")** 01/22 Thu: Word representation: **[03_Word_Representation](assets/slides/CS288_sp26_03_Word_Representation.pdf "assets/slides/CS288_sp26_03_Word_Representation.pdf")** 01/27 Tue: Text classification: **[04_Text Classification](assets/slides/CS288_sp26_04_Text_Classification.pdf "assets/slides/CS288_sp26_04_Text_Classification.pdf")**: **[Assignment 1 released](assignments/Sp2026_CS288_Assignment1.pdf "assignments/Sp2026_CS288_Assignment1.pdf")** 01/29 Thu: Sequence models (Key concepts: Recurrent neural networks): **[05_Sequence Models](assets/slides/CS288_sp26_05_Sequence_Models.pdf "assets/slides/CS288_sp26_05_Sequence_Models.pdf")** 02/03 Tue: Sequence-to-sequence models: **[06_Seq2Seq](assets/slides/CS288_sp26_06_Seq2seq.pdf "assets/slides/CS288_sp26_06_Seq2seq.pdf")** 02/05 Thu: Sequence-to-sequence models (cont’d) & Transformers 02/10 Tue: Transformers (cont’d): **[07_Transformers](assets/slides/CS288_sp26_07_Transformers.pdf "assets/slides/CS288_sp26_07_Transformers.pdf")**: **Assignment 1 due** **Team matching survey due** **Assignment 2 released** 02/12 Thu: Pre-training, Fine-tuning, & Prompting: **[08_Pretraining/FT/Prompting](assets/slides/CS288_sp26_08_Pretraining_Finetuning_Prompting.pdf "assets/slides/CS288_sp26_08_Pretraining_Finetuning_Prompting.pdf")** 02/17 Tue: Pre-training, Fine-tuning, & Prompting (cont’d) 02/19 Thu: Pre-training advanced topics: **[09_Pretraining_Advanced](assets/slides/CS288_sp26_09_Pretraining_Advanced.pdf "assets/slides/CS288_sp26_09_Pretraining_Advanced.pdf")** 02/24 Tue: Post-training: **Assignment 2 due**: **[10_Posttraining](assets/slides/CS288_sp26_10_Posttraining.pdf "assets/slides/CS288_sp26_10_Posttraining.pdf")** 02/26 Thu: Inference methods & Evaluation: **[11_Generation](assets/slides/CS288_sp26_11_Generation.pdf "assets/slides/CS288_sp26_11_Generation.pdf")** 03/03 Tue: Experimental design & Human annotation: **Project Checkpoint 1 (abstract) due** **Assignment 3 released**: **[12_Evaluation_Benchmarking](assets/slides/CS288_sp26_12_Evaluation_Benchmarking.pdf "assets/slides/CS288_sp26_12_Evaluation_Benchmarking.pdf")** 03/05 Thu: Retrieval and RAG: **[13_Retrieval_and_RAG](assets/slides/CS288_sp26_13_Retrieval_and_RAG.pdf "assets/slides/CS288_sp26_13_Retrieval_and_RAG.pdf")** 03/10 Tue: Architecture advanced topics: **[14_Advanced_Architectures](assets/slides/CS288_sp26_14_Advanced_Architectures.pdf "assets/slides/CS288_sp26_14_Advanced_Architectures.pdf")** 03/12 Thu: Impact & Social implications **03/17 Tue**: **No class: EECS faculty retreat** **Assignment 3 early milestone due** 03/19 Thu: Test-time compute & Reasoning models: **Assignment 3 due** **03/24 Tue**: **No class: Spring break** **03/26 Thu**: Office hours 03/31 Tue: Embodied Perception: **[15_Embodied_Perception](assets/slides/CS288_sp26_15_Embodied_Perception.pdf "assets/slides/CS288_sp26_15_Embodied_Perception.pdf")** 04/02 Thu: Inference Time Compute: **[16_Inference_Time](assets/slides/CS288_sp26_16_Inference_Time.pdf "assets/slides/CS288_sp26_16_Inference_Time.pdf")** 04/07 Tue: LLM Reasoning and Agents: **[17_Agent_Reasoning](assets/slides/CS288_sp26_17_Agents_Reasoning.pdf "assets/slides/CS288_sp26_17_Agents_Reasoning.pdf")** 04/09 Thu: Guest lecture: “On the Safety and Security of Computer-Use Agents” by [Huan Sun](https://u.osu.edu/ihudas/people/ "https://u.osu.edu/ihudas/people/") (OSU): **Project Checkpoint 2 (midpoint report) due** 04/14 Tue: Guest lecture: “Memory in Language Models: Representation and Extraction” by [Jack Morris](https://jxmo.io/ "https://jxmo.io/") (Cornell → Stealth) 04/16 Thu: Embodied Agents: **[18_Embodied_Agents](assets/slides/CS288_sp26_18_Embodied_Agents.pdf "assets/slides/CS288_sp26_18_Embodied_Agents.pdf")** 04/21 Tue: Guest lecture: “Continual Learning: Learning during Problem Solving” by [Akshat Gupta](https://akshat57.github.io/ "https://akshat57.github.io/") (UC Berkeley) 04/23 Thu: Guest lecture: “Speech” by [Gopala Anumanchipalli](https://people.eecs.berkeley.edu/~gopala/ "https://people.eecs.berkeley.edu/~gopala/") (UC Berkeley) 04/28 Tue: Project presentation 04/30 Thu: Project presentation: **Project report due by 05/07 (Thu)**

## Acknowledgement

The class materials, including lectures and assignments, are largely based on the following courses, whose instructors have generously made their materials publicly available. We are deeply grateful to them for sharing their work with the broader community:

* [Princeton COS 484 Natural Language Processing](https://princeton-nlp.github.io/cos484/ "https://princeton-nlp.github.io/cos484/") by Danqi Chen, Tri Dao, Vikram Ramaswamy* [CMU Advanced Natural Language Processing](https://cmu-l3.github.io/anlp-fall2025/ "https://cmu-l3.github.io/anlp-fall2025/") by Graham Neubig & Sean Welleck* [Stanford CS336 Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/ "https://stanford-cs336.github.io/spring2025/") by Tatsumori Hashimoto & Percy Liang* [Cornell LM-class](https://lm-class.org/ "https://lm-class.org/") by Yoav Artzi* [An earlier offering of UC Berkeley EECS 288 Natural Language Processing](https://cal-cs288.github.io/fa24/ "https://cal-cs288.github.io/fa24/") by Dan Klein and Alane Suhr

We are grateful to [VESSL AI](https://vessl.ai/ "https://vessl.ai/") and [Google Cloud](https://cloud.google.com/edu/faculty?hl=en "https://cloud.google.com/edu/faculty?hl=en") for providing compute credits to support our final projects.

![VESSL AI](assets/vessl-ai.png) ![Google Cloud](assets/google-cloud.png)