import tkinter as tk
from matplotlib.mathtext import math_to_image
from io import BytesIO
from PIL import ImageTk, Image


class SimpleTableInput(tk.Frame):
    def __init__(self, parent, rows, columns, labels = None):
        tk.Frame.__init__(self, parent)

        self._entry = {}
        self.rows = rows + 1
        self.columns = columns + 1
        if not labels:
            labels = [[row for row in range(rows)], [col for col in range(columns)], 'Index \\ Index']
        elif len(labels) != 3:
            raise ValueError("Please enter your labels in the format [[*xlabels], [*ylabels], '0,0 label']")
        elif rows != len(labels[0]):
            raise ValueError("Your row dimension does not match the number of labels provided")
        elif columns != len(labels[1]):
            raise ValueError('Your column dimension does not match the number of labels provided')

        # register a command to use for validation
        vcmd = (self.register(self._validate), "%P")

        # create the table of widgets
        # TODO add functionality for different types of label i/o
        # TODO work on this, not bad so far!
        for row in range(self.rows):
            for column in range(self.columns):
                index = (row, column)
                if (row and not column) or (not row and column):
                    text = labels[row + 1][column - 1] if not row else labels[column][row - 1]
                    label, e = self.create_label(
                        text
                    ) if '$' not in text else self.createWidgets(text)
                    label.grid(row = row, column = column)
                elif not row and not column:
                    label, e = self.create_label(labels[2])
                    label.grid(row = row, column = column)
                else:
                    e = tk.Entry(self, validate="key", validatecommand=vcmd)
                    e.grid(row=row, column=column, stick="nsew")


                self._entry[index] = e

        # adjust column weights so they all expand equally
        for column in range(self.columns):
            self.grid_columnconfigure(column, weight=1)
        # designate a final, empty row to fill up any extra space
        self.grid_rowconfigure(rows, weight=1)

    def create_label(self, text):
        e = tk.StringVar()
        e.set(text)
        label = tk.Label(self, textvariable=e)
        return label, e

    def createWidgets(self, text):

        #Creating buffer for storing image in memory
        buffer = BytesIO()
        #Writing png image with our rendered greek alpha to buffer
        math_to_image(text, buffer, dpi=1000, format='png')

        #Remoting bufeer to 0, so that we can read from it
        buffer.seek(0)

        # Creating Pillow image object from it
        pimage= Image.open(buffer)

        #Creating PhotoImage object from Pillow image object
        image = ImageTk.PhotoImage(pimage)

        #Creating label with our image
        label = tk.Label(self,image=image)

        #Storing reference to our image object so it's not garbage collected,
        # as TkInter doesn't store references by itself
        label.img = image

        e = 'image'
        return label, e

    def get(self):
        '''Return a list of lists, containing the data in the table'''
        result = []
        for row in range(1,self.rows):
            current_row = []
            for column in range(1,self.columns):
                index = (row, column)

                current_row.append(self._entry[index].get())

            result.append(current_row)
        return result

    def _validate(self, P):
        return
        '''Perform input validation.

        Allow only an empty value, or a value that can be converted to a float
        '''
        if P.strip() == "":
            return True

        try:
            f = float(P)
        except ValueError:
            self.bell()
            return False
        return True

class dh_parameters(tk.Frame):
    #dh_param_labels = ["$"]
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        self.table = SimpleTableInput(self, 3,4, [['1',"2",'3'], [r'$a_{i-1}$', r'$\alpha_{i-1}$', r'$d_{i}$', r'$\theta_{i}$'], 'Joint number \\ DH Param'])
        self.submit = tk.Button(self, text="Submit", command=self.on_submit)
        self.table.pack(side="top", fill="both", expand=True)
        self.submit.pack(side="bottom")
        self.totals = None

    def on_submit(self):
        self.totals = self.table.get()

root = tk.Tk()
a = dh_parameters(root)
a.pack(side="top", fill="both", expand=True)
root.mainloop()
print(a.totals)