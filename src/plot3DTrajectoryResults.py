import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import analyse as an

class Application(tk.Tk):

    def __init__(self, *args, **kwargs):
        """
        Initialise the main application
        """
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Montserrat', size=18)
        self.normal_font = tkfont.Font(family='Montserrat', size=12)
        self.results_file = "No file chosen"

        self.protocol("WM_DELETE_WINDOW", self.on_close)  # Bind close button event

        container = tk.Frame(self, width=1130, height=720)
        container.pack_propagate(False)
        container.pack()

        self.frames = {}
        for F in (StartPage, AnalysisPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''  
        frame = self.frames[page_name]
        frame.tkraise()

    def on_close(self):
        exit()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        """
        Initialise a frame for the home page
        """
        tk.Frame.__init__(self, parent, height=720, width=1130, bg="#ffffff")
        self.controller = controller
        self.pack_propagate(False)

        def choose_file():
            currdir = os.path.join(os.getcwd(), "data")

            controller.results_file = filedialog.askopenfile(parent=self, initialdir=currdir, title='Please Select a Results File:', filetypes=[("Pickle files", "*.pickle")])
            if controller.results_file:
                print("You chose %s" % controller.results_file)
                controller.show_frame("AnalysisPage")

        label_folder = tk.Label(self, text=controller.results_file, font=controller.normal_font, bg="#ffffff")
        label_folder.place(relx=0.5, rely=0.55, anchor="center")
        label = tk.Label(self, text="Home", font=controller.title_font, bg="#ffffff")
        label.place(relx=0.5, rely=0.1, anchor="center")
        button1 = tk.Button(self, text="Choose Results File", command=choose_file)
        button1.place(relx=0.5, rely=0.5, anchor="center")

class AnalysisPage(tk.Frame):

    def __init__(self, parent, controller):
        """
        Initialise a frame for the analyse page
        """
        tk.Frame.__init__(self, parent, height=720, width=1130, background="#ffffff")
        self.controller = controller
        self.pack_propagate(False)
        self.current_frame = 0

        f = plt.figure(figsize=(4, 4), dpi=100)
        a = f.add_subplot(111, projection="3d")
        a.view_init(elev=20., azim=60)

        def rotate_right():
            azimuth = a.azim
            a.view_init(elev=20., azim=azimuth + 10)
            update_canvas()

        def rotate_left():
            azimuth = a.azim
            a.view_init(elev=20., azim=azimuth - 10)
            update_canvas()

        def update_canvas():
            a.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            a.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            a.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

            a.set_xlim3d(3, 8)
            a.set_ylim3d(0, 8)
            a.set_zlim3d(-1.5, 1.5)

            canvas = FigureCanvasTkAgg(f, self)
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            canvas._tkcanvas.place(relx=0.5, rely=0.45, anchor="center")

        def init():
            a.set_xlim3d(-1, 1)
            a.set_ylim3d(26, 28)
            a.set_zlim3d(0, 1)
            a.view_init(elev=20., azim=30)

        def update(i):
            plot_results(i)

        def plot_results(frame=0):
            pose_dict = {}

            results_file = controller.results_file.name
            results_dir = os.path.dirname(results_file)
            frames_dir = os.path.join(results_dir, "frames")

            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)

            results = an.load_pickle(results_file)

            positions = results["positions"]
            print(positions)

            links = [['forehead', 'chin'], ['forehead', 'neck'], ['neck', 'pelvis'], ['neck', 'shoulder1'],
                     ['neck', 'shoulder2'], ['pelvis', 'hip1'], ['pelvis', 'hip2'], ['shoulder1', 'elbow1'],
                     ['shoulder2', 'elbow2'], ['elbow1', 'wrist1'], ['elbow2', 'wrist2'], ['hip1', 'hip2'],
                     ['hip1', 'knee1'], ['hip2', 'knee2'], ['knee1', 'ankle1'], ['knee2', 'ankle2']]

            markers = ["forehead", "chin", "neck", "shoulder1", "elbow1", "wrist1", "shoulder2", "elbow2",
                       "wrist2", "pelvis", "hip1", "hip2", "knee1", "ankle1", "knee2", "ankle2"]

            for i in range(len(markers)):
                point = [positions[frame][i][0], positions[frame][i][1], positions[frame][i][2]]
                pose_dict[markers[i]] = point
                a.scatter(point[0], point[1], point[2], color="red")

            for link in links:
                if len(link) > 1:
                    a.plot3D([pose_dict[link[0]][0], pose_dict[link[1]][0]],
                              [pose_dict[link[0]][1], pose_dict[link[1]][1]],
                              [pose_dict[link[0]][2], pose_dict[link[1]][2]], color="black")

            a.set_xlabel('x')
            a.set_ylabel('y')
            a.set_zlabel('z')

            update_canvas()
            plt.savefig(os.path.join(frames_dir, "img") + str(self.current_frame) + ".jpg", dpi=100)

        def next_frame():
            self.current_frame += 1
            a.clear()
            plot_results(self.current_frame)

        def prev_frame():
            self.current_frame -= 1
            a.clear()
            plot_results(self.current_frame)

        def play_animation():
            ani = FuncAnimation(f, update, 19, interval=40, blit=True)
            writer = PillowWriter(fps=25)
            ani.save("test.gif", writer=writer)

        update_canvas()

        button_next = tk.Button(self, text="Next", command=next_frame)
        button_next.place(relx=0.8, rely=0.3, anchor="center")
        button_prev = tk.Button(self, text="Prev", command=prev_frame)
        button_prev.place(relx=0.8, rely=0.4, anchor="center")

        button_right = tk.Button(self, text="-->", command=rotate_right)
        button_right.place(relx=0.3, rely=0.3, anchor="center")
        button_left = tk.Button(self, text="<--", command=rotate_left)
        button_left.place(relx=0.3, rely=0.4, anchor="center")

        button_anim = tk.Button(self, text="Animation", command=play_animation)
        button_anim.place(relx=0.5, rely=0.8, anchor="center")

        label = tk.Label(self, text="Analyse", font=controller.title_font, background="#ffffff")
        label.place(relx=0, rely=0)

if __name__ == "__main__":
    app = Application()
    app.geometry("1280x720")
    app.title("Final Year Project")
    app.mainloop()
