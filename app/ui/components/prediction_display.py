import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict


class PredictionDisplay(ttk.Frame):
    def __init__(self, parent, language_processor=None):
        """
        Enhanced prediction display panel for Sign Language Recognition.

        Features:
        - Real-time prediction visualization
        - Word and sentence building
        - Confidence feedback with color coding
        - Interactive controls
        - Grammar feedback (optional)

        Args:
            parent: Parent widget (typically the main application window)
            language_processor: Optional grammar checking class with `check_grammar(text)` method
        """
        super().__init__(parent, style='Prediction.TFrame')
        self.language_processor = language_processor
        self.sentence = []          # List of completed words
        self.current_word = []      # Characters in current word
        self.last_prediction = None # Stores last prediction data

        # Configure layout weights
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Main container frame
        container = ttk.Frame(self, padding=10)
        container.grid(row=0, column=0, sticky="nsew")

        # ===== Current Prediction Section =====
        current_frame = ttk.Frame(container)
        current_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(current_frame, text="Current Prediction:",
                  font=('Helvetica', 12)).pack(side=tk.LEFT)

        self.current_symbol = ttk.Label(
            current_frame,
            text="-",
            font=('Helvetica', 18, 'bold'),
            foreground="#3498db",
            width=3
        )
        self.current_symbol.pack(side=tk.LEFT, padx=5)

        self.confidence_label = ttk.Label(
            current_frame,
            text="(0.00)",
            font=('Helvetica', 12)
        )
        self.confidence_label.pack(side=tk.LEFT)

        self.confidence_meter = ttk.Progressbar(
            container,
            orient='horizontal',
            length=200,
            mode='determinate',
            style='Confidence.Horizontal.TProgressbar'
        )
        self.confidence_meter.grid(row=1, column=0, pady=(0, 20), sticky="ew")

        # ===== Word Building Section =====
        word_frame = ttk.Frame(container)
        word_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(word_frame, text="Current Word:",
                  font=('Helvetica', 12)).pack(side=tk.LEFT)

        self.word_display = ttk.Label(
            word_frame,
            text="",
            font=('Helvetica', 16),
            foreground="#2ecc71",
            width=15,
            anchor="w"
        )
        self.word_display.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # ===== Sentence Building Section =====
        sentence_frame = ttk.Frame(container)
        sentence_frame.grid(row=3, column=0, sticky="ew", pady=(0, 20))

        ttk.Label(sentence_frame, text="Sentence:",
                  font=('Helvetica', 12)).pack(side=tk.LEFT)

        self.sentence_display = ttk.Label(
            sentence_frame,
            text="",
            font=('Helvetica', 14),
            wraplength=350,
            justify=tk.LEFT,
            anchor="w"
        )
        self.sentence_display.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # ===== Grammar Feedback Section =====
        self.grammar_label = ttk.Label(
            container,
            text="",
            font=('Helvetica', 10),
            foreground="#e74c3c",
            wraplength=350,
            justify=tk.LEFT
        )
        self.grammar_label.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        # ===== Control Buttons =====
        button_frame = ttk.Frame(container)
        button_frame.grid(row=5, column=0, sticky="ew")

        self.add_space_btn = ttk.Button(
            button_frame,
            text="Add Space",
            command=self.add_space,
            style='Accent.TButton'
        )
        self.add_space_btn.pack(side=tk.LEFT, padx=5, expand=True)

        self.clear_word_btn = ttk.Button(
            button_frame,
            text="Clear Word",
            command=self.clear_word
        )
        self.clear_word_btn.pack(side=tk.LEFT, padx=5, expand=True)

        self.clear_all_btn = ttk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_all
        )
        self.clear_all_btn.pack(side=tk.LEFT, padx=5, expand=True)

        self._configure_styles()

    def _configure_styles(self):
        style = ttk.Style()
        style.configure('Prediction.TFrame', background='#ffffff')
        style.configure('Accent.TButton',
                        foreground='white',
                        background='#3498db',
                        font=('Helvetica', 10, 'bold'))
        style.map('Accent.TButton',
                  background=[('active', '#2980b9'), ('pressed', '#2980b9')])
        style.configure('Confidence.Horizontal.TProgressbar',
                        thickness=20,
                        troughcolor='#ecf0f1',
                        background='#2ecc71',
                        troughrelief='flat',
                        borderwidth=0)

    def update_prediction(self, prediction: Optional[Dict]):
        self.last_prediction = prediction

        if not prediction:
            self.current_symbol.config(text="-", foreground="gray")
            self.confidence_label.config(text="(0.00)")
            self.confidence_meter['value'] = 0
            return

        char = prediction['class']
        confidence = prediction['confidence']

        self.current_symbol.config(text=char)
        self.confidence_label.config(text=f"({confidence:.2f})")
        self.confidence_meter['value'] = confidence * 100

        # Color based on confidence
        if confidence > 0.85:
            color = "#2ecc71"
        elif confidence > 0.7:
            color = "#f39c12"
        else:
            color = "#e74c3c"
        self.current_symbol.config(foreground=color)

        # Append character to word
        if char not in [' ', '<space>', '<del>']:
            self.current_word.append(char)
            self.word_display.config(text=''.join(self.current_word))

        # Grammar check
        if self.language_processor and self.sentence:
            issues = self.language_processor.check_grammar(' '.join(self.sentence))
            if issues:
                self.grammar_label.config(
                    text="\n".join(issues),
                    foreground="#e74c3c"
                )
            else:
                self.grammar_label.config(
                    text="✓ Grammar looks good.",
                    foreground="#2ecc71"
                )

    def add_space(self):
        if self.current_word:
            self.sentence.append(''.join(self.current_word))
            self.current_word = []
            self.word_display.config(text="")
            self._update_sentence_display()

    def clear_word(self):
        self.current_word = []
        self.word_display.config(text="")

    def clear_all(self):
        self.current_word = []
        self.sentence = []
        self.word_display.config(text="")
        self.sentence_display.config(text="")
        self.grammar_label.config(text="")

    def _update_sentence_display(self):
        self.sentence_display.config(text=' '.join(self.sentence))

    def get_current_sentence(self) -> str:
        return ' '.join(self.sentence)

    def get_last_prediction(self) -> Optional[Dict]:
        return self.last_prediction
