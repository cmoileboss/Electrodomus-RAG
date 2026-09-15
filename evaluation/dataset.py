"""Jeu de questions de test pour l'évaluation du retrieval RAG.

Pour chaque question, `relevant_chunk_ids` contient les ids des chunks
(table `chunks`, colonne `id`) considérés comme pertinents pour y répondre.
Ces questions et ids ont été construits à partir des procédures SAV réellement
présentes en base (documents PROC-FR-*/PROC-LV-*/PROC-WX-*, un document = une
procédure ciblée, donc tous ses chunks répondent à la question). Sauf première question.
"""

EVAL_QUESTIONS: list[dict] = [
    {
        "question": "Que faire en cas de code E52 sur WX-500 ?",
        "relevant_chunk_ids": [3]
    },
    {
        "question": "Quelles vérifications faire par téléphone si un four de la gamme FR ne chauffe pas ou cuit mal ?",
        "relevant_chunk_ids": [493, 494, 495, 496, 497],
    },
    {
        "question": "Comment remplacer l'ampoule d'un four et contrôler le joint de porte ?",
        "relevant_chunk_ids": [498, 499, 500, 501, 502, 503],
    },
    {
        "question": "Comment diagnostiquer des traces blanches ou un voile sur les verres en sortie de lave-vaisselle ?",
        "relevant_chunk_ids": [504, 505, 506, 507, 508, 509, 510],  # PROC-LV-01_Traces_blanches_verres
    },
    {
        "question": "Que faire en cas de fuite au sol et de code d'erreur E15 sur un lave-vaisselle ?",
        "relevant_chunk_ids": [511, 512, 513, 514, 515, 516, 517],  # PROC-LV-02_Fuite_E15
    },
    {
        "question": "Quel arbre de diagnostic suivre si la vaisselle ressort mal lavée d'un lave-vaisselle ?",
        "relevant_chunk_ids": [518, 519, 520, 521, 522],  # PROC-LV-03_Vaisselle_mal_lavee
    },
    {
        "question": "Comment guider un client dans le détartrage préventif ou curatif d'un lave-linge ?",
        "relevant_chunk_ids": [523, 524, 525, 526, 527],  # PROC-WX-01_Detartrage
    },
    {
        "question": "Comment diagnostiquer un problème de vidange sur un lave-linge affichant les codes E20 ou E23 ?",
        "relevant_chunk_ids": [528, 529, 530, 531, 532, 533],  # PROC-WX-02_Diagnostic_vidange
    },
    {
        "question": "Comment nettoyer le filtre de vidange d'un lave-linge ?",
        "relevant_chunk_ids": [534, 535, 536, 537, 538, 539],  # PROC-WX-03_Nettoyage_filtre_vidange
    },
    {
        "question": "Que faire en cas de vibrations excessives liées aux brides de transport d'un lave-linge ?",
        "relevant_chunk_ids": [540, 541, 542, 543, 544],  # PROC-WX-04_Brides_transport
    },
]

