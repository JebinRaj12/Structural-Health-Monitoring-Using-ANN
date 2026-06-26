import os
import json
import logging
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import customtkinter as ctk
import matplotlib
# Ensure matplotlib uses the TkAgg backend for embedding in Tkinter
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import config
from predict import SHMPredictor
from utils.pdf_generator import generate_pdf_report

logger = logging.getLogger("SHM_GUI")

class SHMApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure app window settings
        self.title(config.WINDOW_TITLE)
        self.geometry(config.WINDOW_SIZE)
        ctk.set_appearance_mode(config.THEME_MODE)
        ctk.set_default_color_theme(config.COLOR_THEME)
        
        # Initialize variables
        self.predictor = SHMPredictor()
        self.selected_file_path = None
        self.prediction_results = None
        self.canvas = None  # Reference for embedded Matplotlib chart
        
        # Build UI layout
        self._setup_grid()
        self._create_left_panel()
        self._create_right_panel()
        
        # Load and display model performance metrics
        self._load_and_display_metrics()
        
        # Set initial status
        self.set_status("Ready. Please upload a sensor data file to begin analysis.")
        
    def _setup_grid(self):
        """Sets up the main layout grids."""
        # 1 row, 2 columns (33% and 67%)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=33)  # Left panel weight
        self.grid_columnconfigure(1, weight=67)  # Right panel weight
        
        # Left Panel Container
        self.left_container = ctk.CTkFrame(self, fg_color=config.COLOR_BG, corner_radius=0)
        self.left_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        # Right Panel Container
        self.right_container = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=0)
        self.right_container.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        
    def _create_left_panel(self):
        """Builds the Left Panel for data upload and control actions."""
        # Configure grid for left container
        self.left_container.grid_rowconfigure(9, weight=1)  # spacer row pushes logs to bottom
        self.left_container.grid_columnconfigure(0, weight=1)
        
        # Panel Title
        title_label = ctk.CTkLabel(
            self.left_container, 
            text="DATA UPLOAD & CONTROL", 
            font=ctk.CTkFont(family="Helvetica", size=18, weight="bold"),
            text_color=config.COLOR_TEXT
        )
        title_label.grid(row=0, column=0, sticky="w", padx=20, pady=(25, 5))
        
        subtitle_label = ctk.CTkLabel(
            self.left_container, 
            text="Upload CSV or Excel sensor logs to classify", 
            font=ctk.CTkFont(family="Helvetica", size=12),
            text_color=config.COLOR_MUTED
        )
        subtitle_label.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 20))
        
        # Divider
        div = ctk.CTkFrame(self.left_container, height=2, fg_color="#BDC3C7")
        div.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))
        
        # Upload Card
        self.upload_card = ctk.CTkFrame(self.left_container, fg_color=config.COLOR_CARD_BG, border_width=1, border_color="#BDC3C7")
        self.upload_card.grid(row=3, column=0, sticky="ew", padx=20, pady=0)
        self.upload_card.grid_columnconfigure(0, weight=1)
        
        card_label = ctk.CTkLabel(
            self.upload_card, 
            text="Step 1: Select Sensor File", 
            font=ctk.CTkFont(family="Helvetica", size=13, weight="bold"),
            text_color=config.COLOR_TEXT
        )
        card_label.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))
        
        self.browse_btn = ctk.CTkButton(
            self.upload_card, 
            text="Browse CSV / Excel File", 
            font=ctk.CTkFont(family="Helvetica", size=13, weight="bold"),
            command=self.browse_file,
            height=40
        )
        self.browse_btn.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 15))
        
        self.file_name_lbl = ctk.CTkLabel(
            self.upload_card, 
            text="No file selected", 
            font=ctk.CTkFont(family="Helvetica", size=11, slant="italic"),
            text_color=config.COLOR_MUTED,
            wraplength=350
        )
        self.file_name_lbl.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 10))
        
        self.records_count_lbl = ctk.CTkLabel(
            self.upload_card, 
            text="Number of Records: --", 
            font=ctk.CTkFont(family="Helvetica", size=12, weight="bold"),
            text_color=config.COLOR_TEXT
        )
        self.records_count_lbl.grid(row=3, column=0, sticky="w", padx=15, pady=(0, 15))
        
        # Action Card
        self.action_card = ctk.CTkFrame(self.left_container, fg_color=config.COLOR_CARD_BG, border_width=1, border_color="#BDC3C7")
        self.action_card.grid(row=4, column=0, sticky="ew", padx=20, pady=20)
        self.action_card.grid_columnconfigure(0, weight=1)
        
        action_label = ctk.CTkLabel(
            self.action_card, 
            text="Step 2: Run ANN Diagnostics", 
            font=ctk.CTkFont(family="Helvetica", size=13, weight="bold"),
            text_color=config.COLOR_TEXT
        )
        action_label.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))
        
        self.analyze_btn = ctk.CTkButton(
            self.action_card, 
            text="Analyze Structure", 
            fg_color=config.COLOR_PRIMARY,
            state="disabled",
            font=ctk.CTkFont(family="Helvetica", size=13, weight="bold"),
            command=self.run_analysis,
            height=45
        )
        self.analyze_btn.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 12))
        
        self.clear_btn = ctk.CTkButton(
            self.action_card, 
            text="Clear Dashboard", 
            fg_color="#95A5A6",
            hover_color="#7F8C8D",
            font=ctk.CTkFont(family="Helvetica", size=13, weight="bold"),
            command=self.clear_data,
            height=35
        )
        self.clear_btn.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 15))
        
        # Model Performance Box (in Left Panel to organize space)
        self.perf_card = ctk.CTkFrame(self.left_container, fg_color=config.COLOR_CARD_BG, border_width=1, border_color="#BDC3C7")
        self.perf_card.grid(row=5, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.perf_card.grid_columnconfigure((0, 1), weight=1)
        
        perf_title = ctk.CTkLabel(
            self.perf_card, 
            text="Trained Model Validation Metrics", 
            font=ctk.CTkFont(family="Helvetica", size=13, weight="bold"),
            text_color=config.COLOR_TEXT
        )
        perf_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(15, 10))
        
        # Labels for performance metrics
        self.lbl_val_accuracy = self._create_metric_widget(self.perf_card, "Model Accuracy", "row_1", 1)
        self.lbl_val_precision = self._create_metric_widget(self.perf_card, "Precision Score", "row_2", 2)
        self.lbl_val_recall = self._create_metric_widget(self.perf_card, "Recall Score", "row_3", 3)
        self.lbl_val_f1 = self._create_metric_widget(self.perf_card, "F1-Score (Macro)", "row_4", 4)
        
        # Status Log Box
        self.status_frame = ctk.CTkFrame(self.left_container, fg_color="#ECEFF1", corner_radius=0)
        self.status_frame.grid(row=10, column=0, sticky="ew", padx=0, pady=0)
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_lbl = ctk.CTkLabel(
            self.status_frame,
            text="System status logs...",
            font=ctk.CTkFont(family="Helvetica", size=11),
            text_color=config.COLOR_TEXT,
            anchor="w",
            padx=15,
            pady=8
        )
        self.status_lbl.grid(row=0, column=0, sticky="ew")
        
    def _create_metric_widget(self, parent, label_text, widget_id, row):
        """Helper to create a two-column metric row inside parent."""
        title = ctk.CTkLabel(
            parent, 
            text=label_text, 
            font=ctk.CTkFont(family="Helvetica", size=11),
            text_color=config.COLOR_MUTED
        )
        title.grid(row=row, column=0, sticky="w", padx=15, pady=4)
        
        value = ctk.CTkLabel(
            parent, 
            text="--", 
            font=ctk.CTkFont(family="Helvetica", size=12, weight="bold"),
            text_color=config.COLOR_TEXT
        )
        value.grid(row=row, column=1, sticky="e", padx=15, pady=4)
        return value

    def _create_right_panel(self):
        """Builds the Right Panel for showing analysis results."""
        self.right_container.grid_rowconfigure(5, weight=1)  # Bottom row gets charts
        self.right_container.grid_columnconfigure(0, weight=1)
        
        # Dashboard Title
        db_title = ctk.CTkLabel(
            self.right_container, 
            text="STRUCTURAL HEALTH MONITORING DASHBOARD", 
            font=ctk.CTkFont(family="Helvetica", size=20, weight="bold"),
            text_color=config.COLOR_TEXT
        )
        db_title.grid(row=0, column=0, sticky="w", padx=25, pady=(25, 20))
        
        # Top banner for Overall Health result
        self.result_banner = ctk.CTkFrame(self.right_container, fg_color="#F2F4F4", height=80, corner_radius=8)
        self.result_banner.grid(row=1, column=0, sticky="ew", padx=25, pady=(0, 20))
        self.result_banner.grid_columnconfigure(0, weight=1)
        
        self.overall_health_lbl = ctk.CTkLabel(
            self.result_banner, 
            text="OVERALL HEALTH STATUS: NO DATA LOADED", 
            font=ctk.CTkFont(family="Helvetica", size=18, weight="bold"),
            text_color=config.COLOR_MUTED,
            pady=15
        )
        self.overall_health_lbl.grid(row=0, column=0, sticky="ew")
        
        # Frame holding confidence and statistical summaries
        self.summary_container = ctk.CTkFrame(self.right_container, fg_color="transparent")
        self.summary_container.grid(row=2, column=0, sticky="ew", padx=25, pady=0)
        self.summary_container.grid_columnconfigure((0, 1), weight=1, uniform="group1")
        
        # Column 0: Confidence Card
        self.conf_card = ctk.CTkFrame(self.summary_container, fg_color=config.COLOR_BG, height=130)
        self.conf_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.conf_card.grid_columnconfigure(0, weight=1)
        
        conf_title = ctk.CTkLabel(
            self.conf_card, 
            text="PREDICTION CONFIDENCE", 
            font=ctk.CTkFont(family="Helvetica", size=11, weight="bold"),
            text_color=config.COLOR_MUTED
        )
        conf_title.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 5))
        
        self.conf_val_lbl = ctk.CTkLabel(
            self.conf_card, 
            text="--", 
            font=ctk.CTkFont(family="Helvetica", size=32, weight="bold"),
            text_color=config.COLOR_TEXT
        )
        self.conf_val_lbl.grid(row=1, column=0, sticky="w", padx=15, pady=(0, 5))
        
        self.conf_desc_lbl = ctk.CTkLabel(
            self.conf_card, 
            text="Average confidence for classifications", 
            font=ctk.CTkFont(family="Helvetica", size=11),
            text_color=config.COLOR_MUTED
        )
        self.conf_desc_lbl.grid(row=2, column=0, sticky="w", padx=15, pady=(0, 15))
        
        # Column 1: Counts & Distribution Card
        self.stats_card = ctk.CTkFrame(self.summary_container, fg_color=config.COLOR_BG, height=130)
        self.stats_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.stats_card.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        stats_title = ctk.CTkLabel(
            self.stats_card, 
            text="ANALYSIS BREAKDOWN", 
            font=ctk.CTkFont(family="Helvetica", size=11, weight="bold"),
            text_color=config.COLOR_MUTED
        )
        stats_title.grid(row=0, column=0, columnspan=4, sticky="w", padx=15, pady=(15, 5))
        
        # Mini metrics inside grid
        self.stats_total = self._create_card_stat(self.stats_card, "Total", "0", 0)
        self.stats_healthy = self._create_card_stat(self.stats_card, "Healthy", "0 (0%)", 1, color=config.COLOR_HEALTHY)
        self.stats_minor = self._create_card_stat(self.stats_card, "Minor", "0 (0%)", 2, color=config.COLOR_MINOR)
        self.stats_severe = self._create_card_stat(self.stats_card, "Severe", "0 (0%)", 3, color=config.COLOR_SEVERE)
        
        # Engineering Recommendations Card
        self.rec_card = ctk.CTkFrame(self.right_container, fg_color=config.COLOR_BG)
        self.rec_card.grid(row=3, column=0, sticky="ew", padx=25, pady=20)
        self.rec_card.grid_columnconfigure(0, weight=1)
        
        rec_title = ctk.CTkLabel(
            self.rec_card, 
            text="ENGINEERING RECOMMENDATION & ACTION PLAN", 
            font=ctk.CTkFont(family="Helvetica", size=12, weight="bold"),
            text_color=config.COLOR_MUTED
        )
        rec_title.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 5))
        
        self.rec_text_lbl = ctk.CTkLabel(
            self.rec_card, 
            text="Upload a data file and run diagnostics to receive official recommendations.", 
            font=ctk.CTkFont(family="Helvetica", size=13, weight="bold"),
            text_color=config.COLOR_TEXT,
            wraplength=850,
            justify="left"
        )
        self.rec_text_lbl.grid(row=1, column=0, sticky="w", padx=15, pady=(0, 15))
        
        # Section title for distribution chart
        chart_section_title = ctk.CTkLabel(
            self.right_container, 
            text="VISUAL DISTRIBUTION CHART", 
            font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
            text_color=config.COLOR_TEXT
        )
        chart_section_title.grid(row=4, column=0, sticky="w", padx=25, pady=(5, 5))
        
        # Bottom Chart Frame
        self.chart_frame = ctk.CTkFrame(self.right_container, fg_color="#FFFFFF", border_width=1, border_color="#E5E8E8")
        self.chart_frame.grid(row=5, column=0, sticky="nsew", padx=25, pady=(0, 25))
        self.chart_frame.grid_rowconfigure(0, weight=1)
        self.chart_frame.grid_columnconfigure(0, weight=1)
        
        # Placeholder inside chart frame
        self.chart_placeholder = ctk.CTkLabel(
            self.chart_frame, 
            text="[ Visualization Chart Canvas ]\nPrediction distributions will be rendered here.", 
            font=ctk.CTkFont(family="Helvetica", size=12),
            text_color=config.COLOR_MUTED
        )
        self.chart_placeholder.grid(row=0, column=0)
        
    def _create_card_stat(self, parent, label_text, default_val, col, color=None):
        """Helper to create small stacked statistical indicators in stats_card."""
        if color is None:
            color = config.COLOR_TEXT
            
        val_lbl = ctk.CTkLabel(
            parent, 
            text=default_val, 
            font=ctk.CTkFont(family="Helvetica", size=16, weight="bold"),
            text_color=color
        )
        val_lbl.grid(row=1, column=col, pady=(5, 0))
        
        name_lbl = ctk.CTkLabel(
            parent, 
            text=label_text, 
            font=ctk.CTkFont(family="Helvetica", size=10, weight="bold"),
            text_color=config.COLOR_MUTED
        )
        name_lbl.grid(row=2, column=col, pady=(0, 10))
        
        return val_lbl

    def _load_and_display_metrics(self):
        """Reads model metrics JSON and updates left panel labels."""
        if os.path.exists(config.METRICS_PATH):
            try:
                with open(config.METRICS_PATH, 'r') as f:
                    metrics = json.load(f)
                
                self.lbl_val_accuracy.configure(text=f"{metrics.get('accuracy', 0.0):.2f}%")
                self.lbl_val_precision.configure(text=f"{metrics.get('precision', 0.0):.4f}")
                self.lbl_val_recall.configure(text=f"{metrics.get('recall', 0.0):.4f}")
                self.lbl_val_f1.configure(text=f"{metrics.get('f1_score', 0.0):.4f}")
                logger.info("Loaded and displayed validation metrics in GUI.")
            except Exception as e:
                logger.error(f"Failed to display metrics from JSON: {e}")
        else:
            logger.warning("Metrics file not found. UI statistics will remain empty.")

    def set_status(self, text):
        """Updates the status log bar at the bottom of the Left Panel."""
        self.status_lbl.configure(text=text)
        
    def browse_file(self):
        """Triggers the file browsing dialog and parses basic attributes."""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Supported Formats", "*.csv *.xlsx *.xls"),
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx *.xls")
            ]
        )
        
        if not file_path:
            return
            
        self.selected_file_path = file_path
        file_name = os.path.basename(file_path)
        self.file_name_lbl.configure(text=file_name, text_color=config.COLOR_TEXT)
        
        # Pre-scan file to get row count
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
                
            row_count = len(df)
            self.records_count_lbl.configure(text=f"Number of Records: {row_count}")
            
            # Enable analyze button
            self.analyze_btn.configure(state="normal")
            self.set_status(f"File '{file_name}' loaded successfully. Ready to analyze.")
            
        except Exception as e:
            logger.error(f"Error pre-scanning file {file_path}: {e}")
            messagebox.showerror("Invalid File Format", f"Failed to read the file: {str(e)}")
            self.selected_file_path = None
            self.file_name_lbl.configure(text="No file selected", text_color=config.COLOR_MUTED)
            self.records_count_lbl.configure(text="Number of Records: --")
            self.analyze_btn.configure(state="disabled")

    def run_analysis(self):
        """Callback to execute prediction on the loaded file."""
        if not self.selected_file_path:
            return
            
        self.set_status("Running neural network diagnostic analysis...")
        self.analyze_btn.configure(state="disabled")
        self.browse_btn.configure(state="disabled")
        
        # Run prediction in a separate thread so GUI doesn't freeze
        threading.Thread(target=self._execute_prediction_thread, daemon=True).start()

    def _execute_prediction_thread(self):
        try:
            # Predict
            results = self.predictor.predict(self.selected_file_path)
            self.prediction_results = results
            
            # Update GUI elements safely back in Tkinter main loop
            self.after(0, self._update_ui_with_results)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            self.after(0, lambda: self._handle_prediction_error(e))

    def _handle_prediction_error(self, exception):
        messagebox.showerror("Analysis Error", f"Failed to predict structural health:\n{str(exception)}")
        self.set_status("Analysis failed. See log file for details.")
        self.analyze_btn.configure(state="normal")
        self.browse_btn.configure(state="normal")

    def _update_ui_with_results(self):
        """Populates cards and displays graphs based on prediction outputs."""
        res = self.prediction_results
        
        # 1. Update Result Banner (Color codes)
        health = res['overall_health']
        if health == "Healthy":
            bg_col = config.COLOR_HEALTHY
        elif health == "Minor Damage":
            bg_col = config.COLOR_MINOR
        else:
            bg_col = config.COLOR_SEVERE
            
        self.result_banner.configure(fg_color=bg_col)
        self.overall_health_lbl.configure(
            text=f"OVERALL STRUCTURAL HEALTH: {health.upper()}", 
            text_color="#FFFFFF"
        )
        
        # 2. Update Confidence Value
        self.conf_val_lbl.configure(text=f"{res['overall_confidence']:.2f}%")
        
        # 3. Update Breakdown counts & distribution percentages
        self.stats_total.configure(text=str(res['total_samples']))
        self.stats_healthy.configure(text=f"{res['healthy_count']} ({res['healthy_pct']:.1f}%)")
        self.stats_minor.configure(text=f"{res['minor_count']} ({res['minor_pct']:.1f}%)")
        self.stats_severe.configure(text=f"{res['severe_count']} ({res['severe_pct']:.1f}%)")
        
        # 4. Update Engineering Recommendations Card
        recs = config.RECOMMENDATIONS.get(health, [])
        formatted_recs = "\n".join([f"• {r}" for r in recs])
        self.rec_text_lbl.configure(text=formatted_recs)
        
        # 5. Embed Live Matplotlib Pie Chart
        self._embed_matplotlib_chart()
        
        # 6. Automatically generate PDF report
        try:
            pdf_path = generate_pdf_report(res, self.selected_file_path)
            self.set_status(f"Diagnostics complete. Report saved to: {os.path.basename(pdf_path)}")
            # Show a success dialog with path
            messagebox.showinfo(
                "Diagnostics Complete", 
                f"Analysis completed successfully!\n\n"
                f"Overall Structural Health: {health}\n"
                f"Confidence: {res['overall_confidence']:.2f}%\n\n"
                f"PDF report generated in:\n{pdf_path}"
            )
        except Exception as pdf_err:
            logger.error(f"Failed to generate PDF on completion: {pdf_err}")
            messagebox.showwarning("Report Generation Error", f"Analysis completed, but PDF report generation failed:\n{str(pdf_err)}")
            self.set_status("Diagnostics complete, but PDF report failed to save.")
            
        # Re-enable inputs
        self.browse_btn.configure(state="normal")
        self.analyze_btn.configure(state="normal")

    def _embed_matplotlib_chart(self):
        """Draws the prediction pie chart and integrates it into Tkinter."""
        # Remove placeholder and previous canvas
        if self.chart_placeholder:
            self.chart_placeholder.destroy()
            self.chart_placeholder = None
            
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        res = self.prediction_results
        
        # Create a Matplotlib Figure
        fig = Figure(figsize=(7, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        labels = ['Healthy', 'Minor Damage', 'Severe Damage']
        counts = [res['healthy_count'], res['minor_count'], res['severe_count']]
        colors = [config.COLOR_HEALTHY, config.COLOR_MINOR, config.COLOR_SEVERE]
        
        # Filter 0 counts
        f_labels = []
        f_counts = []
        f_colors = []
        for l, c, col in zip(labels, counts, colors):
            if c > 0:
                f_labels.append(f"{l} ({c})")
                f_counts.append(c)
                f_colors.append(col)
                
        if len(f_counts) > 0:
            ax.pie(
                f_counts, 
                labels=f_labels, 
                colors=f_colors, 
                autopct='%1.1f%%', 
                startangle=140,
                textprops={'fontsize': 9, 'weight': 'bold'}
            )
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        else:
            ax.text(0.5, 0.5, "No Data Predicted", ha='center', va='center')
            ax.axis('off')
            
        fig.tight_layout()
        
        # Create Tkinter canvas and draw
        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def clear_data(self):
        """Resets the entire GUI display to empty state."""
        self.selected_file_path = None
        self.prediction_results = None
        
        # Reset file labels
        self.file_name_lbl.configure(text="No file selected", text_color=config.COLOR_MUTED)
        self.records_count_lbl.configure(text="Number of Records: --")
        self.analyze_btn.configure(state="disabled")
        
        # Reset banner
        self.result_banner.configure(fg_color="#F2F4F4")
        self.overall_health_lbl.configure(
            text="OVERALL HEALTH STATUS: NO DATA LOADED", 
            text_color=config.COLOR_MUTED
        )
        
        # Reset confidence & stats
        self.conf_val_lbl.configure(text="--")
        self.stats_total.configure(text="0")
        self.stats_healthy.configure(text="0 (0%)")
        self.stats_minor.configure(text="0 (0%)")
        self.stats_severe.configure(text="0 (0%)")
        
        # Reset recommendations
        self.rec_text_lbl.configure(text="Upload a data file and run diagnostics to receive official recommendations.")
        
        # Reset Matplotlib Canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
            
        # Recreate placeholder
        self.chart_placeholder = ctk.CTkLabel(
            self.chart_frame, 
            text="[ Visualization Chart Canvas ]\nPrediction distributions will be rendered here.", 
            font=ctk.CTkFont(family="Helvetica", size=12),
            text_color=config.COLOR_MUTED
        )
        self.chart_placeholder.grid(row=0, column=0)
        
        self.set_status("Dashboard cleared. Ready for new files.")
        logger.info("Cleared GUI dashboard metrics.")
        
if __name__ == "__main__":
    app = SHMApp()
    app.mainloop()
