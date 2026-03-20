Elective - III
UE23AM342BA3 : Explainable AI
Banana Problem - Unit 3
Prof. Ayisha Noori V K
Department of Computer Science and Engineering (AI&ML)

# Banana Problem - Unit3 Individual Activity (10 Marks)

## For Odd SRNs

Complete Task 1 and Task 2.

## Task 1: Vanilla Gradient Saliency Map for Banana Classification

Using a pretrained CNN (e.g., ResNet18) trained on ImageNet:

1. Load an image of a banana.
2. Compute the Vanilla Gradient saliency map for the banana class.
3. Visualize the saliency map.
4. Comment on whether edges or object regions are highlighted.

## Task 2: Smooth Visualization via Activation Maximization

Generate a synthetic image that maximally activates the banana class neuron in ResNet18 using gradient ascent:

1. Start from random noise.
2. Optimize the input to maximize banana class score.
3. Visualize the generated pattern.

## Expected Interpretation

### Task 1
- Mostly highlights edges and contours
- Often noisy due to gradient saturation
- Background edges may also appear

### Task 2
- Generates yellow curved textures
- Shows texture bias of CNN
- May look noisy without regularization

## Submission Link

https://forms.gle/anBV6QAur9BkdbTk8
