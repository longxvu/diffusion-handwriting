import matplotlib.pyplot as plt
import sys

STEPS_PER_EPOCH = 306

def parse_file(f):
    epoch, stroke_loss, drawn_loss, total_loss = [], [], [], []
    epoch_count = 0
    avg_stroke, avg_drawn = 0, 0
    for step, line in enumerate(f):
        s_l, p_l = [float(x) for x in line.split()]
        avg_stroke += s_l
        avg_drawn += p_l
        if step != 0 and step % (STEPS_PER_EPOCH-1) == 0:
            epoch.append(epoch_count)
            stroke_loss.append(avg_stroke / STEPS_PER_EPOCH)
            drawn_loss.append(avg_drawn / STEPS_PER_EPOCH)
            total_loss.append((avg_stroke + avg_drawn) / STEPS_PER_EPOCH)
            epoch_count += 1
            avg_stroke, avg_drawn = 0, 0
    return epoch, stroke_loss, drawn_loss, total_loss

def generate_axis(title, y_label):
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)

if __name__ == "__main__":
   filename = sys.argv[1] 
   with open(filename) as f:
       steps, stroke_loss, drawn_loss, total_loss = parse_file(f)
   print(steps, stroke_loss, drawn_loss, total_loss)

   plt.plot(steps, stroke_loss)
   generate_axis("Stroke Loss over Training Epochs", "Stroke Loss")
   plt.savefig("stroke_loss")
   plt.clf()
   generate_axis("Draw Loss over Training Epochs", "Draw Loss")
   plt.plot(steps, drawn_loss)
   plt.savefig("drawn_loss")
   plt.clf()
   generate_axis("Total Loss over Training Epochs", "Total Loss")
   plt.plot(steps, total_loss)
   plt.savefig("total_loss")
   plt.clf()
