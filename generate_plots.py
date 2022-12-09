import matplotlib.pyplot as plt
import sys

def parse_file(f):
    steps, stroke_loss, drawn_loss, total_loss = [], [], [], []
    for step, line in enumerate(f):
        steps.append(step)
        s_l, p_l = [float(x) for x in line.split()]
        stroke_loss.append(s_l)
        drawn_loss.append(p_l)
        total_loss.append(s_l + p_l)
    return steps, stroke_loss, drawn_loss, total_loss

def generate_axis(title, y_label):
    plt.xlabel("Steps")
    plt.ylabel(y_label)
    plt.title(title)

if __name__ == "__main__":
   filename = sys.argv[1] 
   with open(filename) as f:
       steps, stroke_loss, drawn_loss, total_loss = parse_file(f)

   plt.plot(steps, stroke_loss)
   generate_axis("Stroke Loss over Training Steps", "Stroke Loss")
   plt.savefig("stroke_loss")
   plt.clf()
   generate_axis("Draw Loss over Training Steps", "Draw Loss")
   plt.plot(steps, drawn_loss)
   plt.savefig("drawn_loss")
   plt.clf()
   generate_axis("Total Loss over Training Steps", "Total Loss")
   plt.plot(steps, total_loss)
   plt.savefig("total_loss")
   plt.clf()
