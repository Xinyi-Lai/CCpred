import tkinter as tk
from tkinter import ttk, scrolledtext
import sys
from io import StringIO

class App:
    def __init__(self, master):
        self.master = master
        master.title('基于混合模型的短期碳交易价格预测软件')
        master.geometry('650x550')
        
        # Create and add input fields
        self.create_input_fields()

        # Add a button to trigger API call
        self.add_api_button()
        self.output_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=50, height=30)
        self.output_text.grid(row=0, column=2, rowspan=23, padx=10, pady=10)

    def create_input_fields(self):

        # market
        ttk.Label(self.master, text="选择需要预测的市场:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(self.master, values=['EU', 'GZ', 'HB', 'SH', 'BJ', 'FJ', 'CQ', 'TJ', 'SZ'], state="readonly").grid(row=0, column=1)

        # # Sample Rate
        # ttk.Label(self.master, text="Data Sample Rate:").grid(row=0, column=0, sticky="w")
        # self.sample_rate_entry = ttk.Entry(self.master)
        # self.sample_rate_entry.grid(row=1, column=1)

        # Appliances
        ttk.Label(self.master, text="Appliances:").grid(row=2, column=0, sticky="w")
        self.appliances_entry = ttk.Entry(self.master)
        self.appliances_entry.grid(row=3, column=1)

        # Data
        ttk.Label(self.master, text="Data Path:").grid(row=4, column=0, sticky="w")
        self.data_path_entry = ttk.Entry(self.master)
        self.data_path_entry.grid(row=5, column=1)

        # Train Method
        ttk.Label(self.master, text="Method").grid(row=6, column=0, sticky="w")
        self.method_entry = ttk.Entry(self.master)
        self.method_entry.grid(row=7, column=1)

        # sequence_length;n_epochs;batch_size
        ttk.Label(self.master, text="Parameters").grid(row=8, column=0, sticky="w")
        ttk.Label(self.master, text="sequence_length").grid(row=8, column=1, sticky="w")
        ttk.Label(self.master, text="n_epochs").grid(row=10, column=1, sticky="w")
        ttk.Label(self.master, text="batch_size").grid(row=12, column=1, sticky="w")
        self.sequence_length_entry = ttk.Entry(self.master)
        self.sequence_length_entry.grid(row=9, column=1)
        self.n_epochs_entry = ttk.Entry(self.master)
        self.n_epochs_entry.grid(row=11, column=1)
        self.batch_size_entry = ttk.Entry(self.master)
        self.batch_size_entry.grid(row=13, column=1)

        # Train Start Time
        ttk.Label(self.master, text="Train Start Time:").grid(row=14, column=0, sticky="w")
        self.train_start_time_entry = ttk.Entry(self.master)
        self.train_start_time_entry.grid(row=15, column=1)

        # Train End Time
        ttk.Label(self.master, text="Train End Time:").grid(row=16, column=0, sticky="w")
        self.train_end_time_entry = ttk.Entry(self.master)
        self.train_end_time_entry.grid(row=17, column=1)

        # Test Start Time
        ttk.Label(self.master, text="Test Start Time:").grid(row=18, column=0, sticky="w")
        self.test_start_time_entry = ttk.Entry(self.master)
        self.test_start_time_entry.grid(row=19, column=1)

        # Test End Time
        ttk.Label(self.master, text="Test End Time:").grid(row=20, column=0, sticky="w")
        self.test_end_time_entry = ttk.Entry(self.master)
        self.test_end_time_entry.grid(row=21, column=1)

        ttk.Label(self.master, text="Result").grid(row=1, column=2, sticky="n", pady=10, columnspan=2)

    def add_api_button(self):
        button = ttk.Button(self.master, text="Run", command=self.run_api)
        button.grid(row=22, column=1, columnspan=1, pady=10)

    def run_api(self):
        # Collect input values

        sample_rate = int(self.sample_rate_entry.get())
        appliances = self.appliances_entry.get()
        data_path = self.data_path_entry.get()
        method = self.method_entry.get()
        sequence_length = self.sequence_length_entry.get()
        n_epochs = self.n_epochs_entry.get()
        batch_size=self.batch_size_entry.get()
        train_start_time = self.train_start_time_entry.get()
        train_end_time = self.train_end_time_entry.get()
        test_start_time = self.test_start_time_entry.get()
        test_end_time = self.test_end_time_entry.get()
        algorithm_class = globals().get(method, None)

        # Construct the parameter dictionary
        parameters = {
            # Specify power type, sample rate and disaggregated appliance
            'power': {
                'mains': ['active'],
                'appliance': ['active']
            },
            'sample_rate': sample_rate,
            'appliances': [appliances],
            'light': ['light'],
            'sockets': ['sockets'],
            'AHU': ['AHU'],
            'elevator': ['elevator'],
            # Universally no pre-training
            'pre_trained': False,
            # Specify algorithm
            'methods': {"CNN": algorithm_class({'sequence_length': int(sequence_length), 'n_epochs': int(n_epochs), 'batch_size': int(batch_size)})},
            # Specify train and test data
            'train': {
                'datasets': {
                    'combed': {
                        'path':data_path,
                        'buildings': {
                            1: {
                                'start_time': train_start_time,
                                'end_time': train_end_time
                            }
                        }
                    },
                }
            },
            'test': {
                'datasets': {
                    'combed': {
                        'path': data_path,
                        'buildings': {
                            1: {
                                'start_time': test_start_time,
                                'end_time': test_end_time
                            }
                        }
                    },

                },
                # Specify evaluation metrics
                'metrics': ['mae', 'f1score', 'recall', 'precision']
            }
        }

        # 保存原始的 sys.stdout，以便之后恢复
        original_stdout = sys.stdout

        # 创建一个 StringIO 对象，用于捕获 stdout
        stdout_capture = StringIO()
        sys.stdout = stdout_capture

        try:
            # 调用 API
            API(parameters)
        finally:
            # 恢复 sys.stdout，并获取捕获的输出
            sys.stdout = original_stdout
            output_result = stdout_capture.getvalue()

            # 将输出结果显示在 GUI 文本框中
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, output_result)


# Create the main window
root = tk.Tk()

# Create an instance of the application
app = App(root)

# Run the main loop
root.mainloop()