# 🎮 Trackmania Asset Generation (Diffusion-Based)

A personal machine learning project to generate new Trackmania map assets using generative models, starting from existing texture data.

---

## 📌 Project Overview

This project explores **procedural asset generation** using modern generative models, with a focus on:

* Generating new **base textures** from existing assets
* Extending to **multi-map generation**:

  * Base (albedo)
  * Roughness
  * Normal / GL maps
* Eventually enabling **text-to-asset generation**

The long-term vision is to build a pipeline capable of generating coherent, production-ready assets for Trackmania mapping.

---

## 🧠 Problem Framing

Each asset is currently represented by:

* `base` → visual texture (RGB)
* `rough` → roughness map
* `nor_gl` → normal + gloss map

### Objectives

#### Phase 1 (Current)

* Learn distribution of base textures
* Generate new plausible base assets

#### Phase 2

* Generate variations from existing assets (image-to-image)

#### Phase 3

* Generate all maps:

  * base → rough
  * base → normal/gloss

#### Phase 4

* Text-to-image asset generation

  * e.g. *"metallic desert block with scratches"*

---

## 🏗️ Approach

### Why Diffusion Models?

We use diffusion models (e.g. Stable Diffusion) because they:

* Handle **small datasets** better than GANs
* Are **stable to train**
* Support:

  * text-to-image
  * image-to-image
  * conditional generation

---

## ⚙️ Technical Stack

* **Frameworks**

  * PyTorch
  * Hugging Face `diffusers`
  * `peft` (LoRA fine-tuning)

* **Model**

  * Stable Diffusion v1.5 (pretrained)
  * LoRA for efficient adaptation

---

## 📂 Dataset Structure

Initial setup (Phase 1):

```
dataset/
  0001.png
  0002.png
  ...
```

Later (multi-map):

```
dataset/
  asset_001/
    base.png
    rough.png
    nor_gl.png
```

---

## Quickstart
```
TODO
```
---

## 🚀 Training Strategy

### Phase 1 — Base Generation

* Fine-tune Stable Diffusion using **LoRA**
* Input: noise + prompt
* Output: base texture

Prompt (initial):

```
"trackmania asset"
```

---

### Phase 2 — Image-to-Image

* Input: existing asset
* Output: variation
* Goal: expand asset diversity

---

### Phase 3 — Multi-Map Generation

Possible strategies:

1. **Separate models**

   * base generator
   * base → rough
   * base → normal

2. **Conditional pipeline (preferred)**

   * Generate base
   * Derive other maps from base

---

### Phase 4 — Text Conditioning

* Introduce captions
* Enable prompt-based generation

---

## 📊 Evaluation

Since this is a generative task, evaluation is primarily:

* Visual quality
* Style consistency
* Diversity of outputs
* Usability in Trackmania maps

---

## 🧪 Experiments

| Experiment           | Description                       | Status         |
| -------------------- | --------------------------------- | -------------- |
| Base generation      | LoRA fine-tuning on base textures | 🔄 In progress |
| Img2Img variation    | Generate variations from assets   | ⏳ Planned      |
| Multi-map generation | Predict rough/normal from base    | ⏳ Planned      |
| Text-to-asset        | Prompt-based generation           | ⏳ Planned      |

---

## ⚠️ Challenges

* Limited dataset size
* Maintaining texture coherence
* Multi-map consistency (base vs normal vs roughness)
* Avoiding overfitting

---

## 🔮 Future Ideas

* ControlNet for structure-aware generation
* Tileable texture generation
* Latent space interpolation between assets
* Integration into Trackmania editor workflow

---

## 🤝 Contributing

This is a personal experimental project, but ideas and discussions are welcome.

---

## 📜 License

This project is licensed under the MIT License.

You are free to:

Use
Modify
Distribute
Include in commercial or non-commercial projects
Conditions:
Include the original license and copyright notice

See full license text below or in the LICENSE file.

---

## 🙌 Acknowledgements

* Hugging Face Diffusers
* Stable Diffusion community
* Trackmania mapping community

---
