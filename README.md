CANXURE AI --- Precision Oncology Virtual Tumor Research Framework
Research use only. CANXURE is an experimental computational research framework. Its outputs are hypotheses for investigation, not clinical treatment recommendations. They must not be used to select, withhold, dose, or modify treatment for any patient.
<p align="center">
<strong>{=html}Multimodal machine learning for exploring cancer-cell and drug-response relationships</strong>{=html}
</p>
CANXURE (also referred to as CANXURE AI) is a research project exploring how cancer-cell molecular profiles and drug representations can be combined computationally to study anticancer drug responses. The repository contains a modular Python research framework for virtual-tumor simulations, preprocessing, model development, evaluation utilities, and reproducible experimentation.
The project's longer-term research direction is to connect molecular information---such as gene expression and mutation profiles---with chemical representations of drugs and experimentally measured response data. The framework is digital-twin-inspired; this wording does not imply that CANXURE is currently a validated, patient-specific, dynamic, or clinically deployable digital twin.
Contents
1. Project at a glance
2. Research question and intended use
3. Important status distinction: repository demo vs. real-data work
4. Architecture overview
5. ResearchDigitalTwin model
6. Molecular data and drug representations
7. Real-data pipeline and documented dataset statistics
8. Data processing and dimensionality reduction
9. Training, validation, and evaluation
10. Repository layout
11. Environment and installation
12. Running the demonstration workflow
13. Running the research-model smoke-training script
14. Reproducing the real-data pipeline
15. Artifacts and expected outputs
16. Reproducibility and data integrity
17. Current limitations and known open research tasks
18. Responsible interpretation
19. Troubleshooting
20. Roadmap
21. Contributing and research collaboration
22. Data sources and acknowledgements
23. License and citation
1. Project at a glance
Item                                Description
Project                             CANXURE AI
Repository                          battlehubindia81-glitch/Canxure
Domain                              Computational oncology, drug-response modelling, multimodal machine learning
Primary language                    Python
Neural-network framework            PyTorch
Data tooling                        NumPy, pandas, SciPy, scikit-learn
Chemistry tooling                   RDKit-based molecular processing in the real-data workflow
Model concept                       Multimodal encoders, feature fusion, and multiple research-output heads
Research data direction             DepMap molecular profiles + GDSC drug-response measurements
Demonstration mode                  Synthetic/demo inputs; not clinical evidence
Intended users                      Computational biology researchers, ML engineers, bioinformatics learners, and collaborators
Status                              Research prototype; validation and reproducibility work remain necessary
Core idea
CANXURE investigates whether a computational model can learn associations between:
Cancer-cell molecular characteristics, including gene-expression and mutation-derived features.
Drug molecular representations, such as RDKit-derived fingerprints.
Measured experimental response, such as GDSC area-under-the-dose-response-curve (AUC) values.
The key research unit for the documented real-data pipeline is a cell-line--drug pair: a cancer cell line, a compound tested against it, and an experimentally measured response.
What CANXURE is not
At its current research stage, CANXURE should not be described as:
A clinically validated treatment recommendation system.
A replacement for oncologists, molecular tumor boards, or laboratory assays.
A validated predictor of an individual patient's response or toxicity.
A survival-analysis system merely because a model output can be exponentiated.
A graph-neural-network drug model unless the relevant graph construction and GNN layers are present in the version being discussed.
A fully validated patient digital twin.
2. Research question and intended use
Primary research question
Can molecular features of a cancer cell line and a numerical representation of a drug be integrated to estimate the cell line's experimentally observed response to that compound?
This is a supervised-learning problem when each example has an observed response target.
For a cell line (c) and drug (d), the general formulation is:
[ \hat{y}{=tex}{c,d}=f\theta{=tex}(x_c,x_d) ]
where:
(x_c) is the molecular feature representation of the cell line.
(x_d) is the representation of the drug.
(f_\theta{=tex}) is a parameterized model.
(\hat{y}{=tex}_{c,d}) is the predicted outcome.
(y_{c,d}), when available, is the measured experimental target.
The precise meaning and units of the output depend on the target and training objective used in a specific experiment. In the documented real-data work, the response target is GDSC log2 AUC, not patient survival time.
Research objectives
Integrate heterogeneous cell-line and drug-response datasets.
Preserve traceability between source identifiers and processed features.
Reduce very high-dimensional molecular measurements into manageable model inputs.
Generate numerical drug representations from validated chemical structures.
Develop and inspect multimodal neural-network components.
Establish honest baselines and held-out evaluation protocols.
Identify failure modes, data gaps, and generalization limitations.
Create a framework that can be extended through collaboration and further validation.
Scope
The framework is for computational research and experimentation. Any result must be interpreted in the context of the source assay, the specific target definition, preprocessing, model checkpoint, split strategy, and evaluation metrics.
3. Important status distinction: repository demo vs. real-data work
Do not conflate the runnable repository demonstration with the separate real-data research pipeline.
The repository contains a synthetic demonstration workflow. The real-data pipeline was also developed in a Google Colab workflow using external datasets and Drive-backed artifacts. Those are distinct execution paths.
Component                                   What it demonstrates       What it does not prove
python scripts/run_simulation.py --demo   A software demonstration   It does not validate of the                     drug-response virtual-tumor/simulation   prediction on workflow using generated   DepMap/GDSC or patient demo inputs                data
python scripts/train_research.py          A one-step model/training  It is not, by itself, smoke test using randomly  training on the real generated tensors and a    GDSC response table synthetic target
Real-data Colab workflow                    Processing of              It does not establish DepMap/GDSC-derived data   clinical utility or and construction of        reliable predictive molecular feature          performance by itself artifacts, as documented
in the project work
Reported held-out baseline                  An initial evaluation      It should not be result from the documented presented as a real-data work             validated or final performance claim without reproducing the exact experiment
The repository's research training script creates random input tensors and a random target for a minimal optimization step. Its output should be understood as a software smoke test for model construction, forward propagation, gradient computation, checkpointing, and summary generation---not as a scientific benchmark.
Similarly, a demo report or a plausible-looking prediction does not establish that a model learned a clinically meaningful relationship.
Data availability
Large external datasets, derived features, and checkpoints may be stored outside Git, including in Google Drive. A file existing in a personal Colab runtime or Drive folder does not mean it is included in the public repository.
Before attempting to reproduce a result, confirm that the required raw data, processed feature files, identifier mappings, fitted preprocessors, model checkpoint, and configuration are all available and correspond to the same experiment.
4. Architecture overview
At a high level, CANXURE explores a multimodal architecture: different input modalities are processed by dedicated encoders, their representations are combined, and task-specific heads produce research outputs.
Cancer-cell expression ──> Expression encoder ──┐
                                                │
Cancer-cell mutations ───> Mutation encoder ────┤
                                                ├──> Feature fusion
Biology features ─────────> Biology encoder ─────┤        │
                                                │        v
Drug fingerprint ─────────> Drug encoder ────────┘   Shared representation
                                                         │
                     ┌───────────────────────────────────┼────────────────────┐
                     v                                   v                    v
                Efficacy head                       Toxicity head        Resistance head

                                      Clinical-response head (two classes)
This diagram represents the structure of the documented ResearchDigitalTwin class. It does not mean that each output is already trained on a validated, independent biological endpoint.
Conceptual data flow
Input preparation: numerical arrays are prepared in the expected order and dimensions.
Modality-specific encoding: each encoder maps its input into a learned representation.
Concatenation: encoded vectors are concatenated along the feature dimension.
Fusion network: fully connected layers transform the combined representation.
Output heads: separate modules generate efficacy-, toxicity-, resistance-, and clinical-response-related outputs.
Task-specific interpretation: outputs must be interpreted only in accordance with the labels, losses, and validation data used for the particular experiment.
Why multimodal learning?
Gene expression, mutation-derived features, and chemical fingerprints represent different kinds of information. A combined model can learn statistical relationships across these modalities. Whether fusion improves performance must be tested against appropriate single-modality and simple baseline models.
Model family and implementation caution
The documented ResearchDigitalTwin is implemented as a dense neural network with multilayer perceptron (MLP) encoders and fully connected fusion/output layers. Its drug input is a 1,024-dimensional vector. A graph neural network (GNN), Graph Convolutional Network (GCN), or Graph Attention Network (GAT) should only be claimed for a specific model version if the source actually constructs molecular graphs and applies graph-convolution/attention layers.
5. ResearchDigitalTwin model
Implementation: src/models/research_model.py
Input interface
The model expects four batched torch.float32 tensors:
Argument         Expected shape Description
expression         (B, 500) Reduced expression feature vector mutation            (B, 64) Reduced mutation feature vector biology             (B, 64) Additional biology feature vector drug              (B, 1024) Drug fingerprint vector
B is the batch size. For example, a batch of 16 examples uses expression input shape (16, 500).
The rows across the input tensors must refer to the same examples. A row-alignment error can silently associate one cell line's features with a different drug or response target, invalidating the experiment.
Encoder layers
The current class defines the following MLP dimensions:
Module                   Layer dimensions               Encoded output
Expression encoder       500 → 1024 → 512                        512 Mutation encoder         64 → 512 → 256                          256 Biology encoder          64 → 512 → 256                          256 Drug encoder             1024 → 512 → 256                        256 Concatenated vector      512 + 256 + 256 + 256                 1,280 Fusion network           1280 → 768 → 512                        512 Efficacy head            512 → 128 → 1                             1 Toxicity head            512 → 128 → 1                             1 Resistance head          512 → 128 → 1                             1 Clinical-response head   512 → 128 → 2 by default                  2
MLP details
The MLP helper adds the following operations between intermediate linear layers:
nn.Linear
nn.LayerNorm
nn.GELU
nn.Dropout
The final layer in each MLP does not receive those intermediate normalization/activation/dropout layers from the helper. The model uses dropout 0.1 by default.
Layer normalization normalizes intermediate activations within a sample. GELU is a nonlinear activation. Dropout randomly masks activations during training to regularize the network. These components can help optimization and reduce overfitting, but they do not guarantee generalization.
Fusion
The implementation encodes all four inputs and concatenates the resulting representations along dimension 1:
fused = self.fusion(
    torch.cat(
        [
            self.expression_encoder(expression),
            self.mutation_encoder(mutation),
            self.biology_encoder(biology),
            self.drug_encoder(drug),
        ],
        dim=1,
    )
)
The resulting 1,280-dimensional vector is passed through the fusion network to produce a 512-dimensional shared representation.
Output semantics
The model returns a dictionary with these keys:
efficacy: one scalar per example from the efficacy head.
toxicity: one sigmoid-transformed scalar per example.
resistance: one sigmoid-transformed scalar per example.
clinical_response: two softmax-transformed values per example.
Expected output shapes are (B, 1) for the three scalar heads and (B, 2) for clinical_response.
Important: Output transformations do not define the scientific target. The toxicity and resistance heads return sigmoid values, but those values are not automatically calibrated probabilities. Calibration and interpretation depend on the training labels, loss functions, class balance, validation, and calibration procedures. The efficacy head returns a raw scalar; its units depend on the target used for training. The clinical-response head returns a two-class softmax distribution, but meaningful class probabilities require appropriate training and evaluation.
The class's forward method defines predictions; it does not define a complete scientifically valid multi-task training procedure. Each head needs a clearly specified target, appropriate loss, documented sample coverage, and validation.
Biology input dependency
The model requires a 64-dimensional biology input. In the previously documented real-data workflow, a corresponding biology feature file had not yet been located. This remains a reproducibility dependency until a valid feature-generation process and artifact are documented.
Do not fill this input with arbitrary values and describe the resulting output as biologically meaningful. If a run uses a placeholder or synthetic biology input, label it explicitly as a software test.
Model outputs are not clinical endpoints by default
The names efficacy, toxicity, resistance, and clinical_response describe intended output categories in the software. They do not, on their own, prove that the model has been trained with measured efficacy, toxicity, resistance, or clinical outcome labels. Check the experiment configuration and target construction before interpreting any head scientifically.
6. Molecular data and drug representations
SMILES and RDKit
SMILES (Simplified Molecular Input Line Entry System) is a text notation for molecular structures. RDKit can parse valid SMILES, represent molecules, and calculate chemical features.
The documented real-data pipeline used validated drug structures to generate 1,024-dimensional RDKit-derived fingerprint vectors.
Conceptually:
Drug identifier
     |
     v
Validated compound mapping
     |
     v
SMILES string
     |
     v
RDKit molecule parsing and validation
     |
     v
1,024-dimensional fingerprint
     |
     v
Drug encoder in ResearchDigitalTwin
The exact fingerprint algorithm and parameters should be obtained from the relevant feature-generation script and recorded with the artifact. Do not assume every RDKit fingerprint family uses the same encoding.
Why use fingerprints?
Fingerprints provide fixed-length numerical representations that can be consumed by conventional machine-learning models and MLPs. They encode structural patterns according to the selected fingerprint algorithm.
Fingerprints are not a complete representation of pharmacology. They do not, by themselves, capture dose, exposure, metabolism, target engagement, all stereochemical or conformational effects, or the tumor microenvironment.
Identifier mapping
A numeric drug identifier in GDSC should not be assumed to be a PubChem Compound ID (CID). Identifier mapping must use a documented source and should preserve the original identifiers and mapping provenance.
The documented work used a GDSC SMILES mapping resource and RDKit validation. Structures that could not be reliably mapped were left unresolved rather than guessed.
GNNs are a possible extension, not a synonym for fingerprints
A graph-based molecular model would generally require atom nodes, bond edges, node/edge features, graph batching, graph-convolution or attention layers, and a graph readout/pooling operation. That is a different representation path from passing a fixed 1,024-dimensional fingerprint to an MLP.
Any future GNN implementation should be benchmarked against the fingerprint baseline using the same splits and evaluation protocol.
7. Real-data pipeline and documented dataset statistics
The real-data research work integrated DepMap molecular profiles with GDSC drug-response measurements. The statistics below are documented snapshots from the project's processing work, not a guarantee that every artifact is committed to this repository or that every result is currently reproducible from a fresh clone.
Source datasets
DepMap: cancer-cell molecular data, including expression and mutation-derived information.
GDSC: experimental cancer-cell drug-sensitivity measurements, including AUC-based response data.
Drug structure mapping resource: used to associate a subset of GDSC drug identifiers with SMILES structures for RDKit processing.
Use the source release/version and data-use conditions that match the actual downloaded files. Dataset releases can change, so record exact filenames, release identifiers, download dates, checksums, and any filtering decisions in the experiment metadata.
Documented cohort and response counts
Quantity                                               Documented value
Common cell-line cohort across                                      696 the integrated source tables
Unique drugs in the response                                        537 table
Original combined response                                      356,498 measurements before final
consolidation
Measurements in the                                             284,863 consolidation step described
in the project notes
Unique consolidated                                             255,133 cell-line--drug pairs
Pairs with one measurement in                                   225,403 that consolidation step
Pairs with two measurements in                                   29,730 that consolidation step
Duplicate rows removed by                                        29,730 averaging
Drugs with validated mapped                                         421 structures/fingerprints
Drugs without validated mapped                                      116 structures in that mapping
pass
Response measurements covered          284,863 (approximately 79.91% of by mapped drugs in the            measurements in that reported subset) documented mapping pass
The counts above come from different stages of the pipeline. The original combined response table and the later consolidation subset do not have identical row counts; the exact filtering sequence must be followed to reconcile them. Do not present the values as if they all refer to the same intermediate table.
Documented processed artifacts
The previous Colab workflow recorded the following Drive paths:
/content/drive/MyDrive/Canxure_backup/
├── data/
│   ├── raw/
│   │   ├── GDSC1_(log2_AUC)_subsetted.csv
│   │   ├── GDSC2_(log2_AUC)_subsetted.csv
│   │   ├── Damaging_Mutations_(Public_26Q1)_subsetted.csv
│   │   └── Expression_(Short-read)_Public_26Q1_subsetted.csv
│   ├── processed/
│   │   ├── real_consolidated_response_pairs.csv
│   │   ├── real_expression_500d.npz
│   │   └── real_mutation_64d.npz
│   └── features/
│       └── real_gdsc_drug_fingerprints_1024d.npz
These are historical workflow paths, not paths guaranteed to exist on every machine or in the GitHub repository. The files may be large and may be excluded from version control. Follow applicable dataset licenses and access requirements.
Response target
The documented target is the measured GDSC log2 AUC associated with a cell-line--drug pair. AUC summarizes response across the tested concentration range according to the dataset's assay and normalization convention.
Do not describe log2 AUC as survival time, a hazard ratio, clinical efficacy, or a directly observed patient outcome.
A model trained on this target should be evaluated as an AUC-response prediction task unless the target is deliberately changed and the new task is independently documented.
Drug-structure coverage
The documented mapping linked 421 of 537 drug identifiers to validated structures (approximately 78.4% of the unique drugs). The remaining 116 were not assigned invented SMILES.
This matters because response rows for drugs without valid structures cannot be used by a pipeline that requires the corresponding fingerprint unless a defensible alternative representation is defined. Report both row-level coverage and unique-drug coverage; these are different measures.
8. Data processing and dimensionality reduction
Expression
The source expression matrix had approximately 19,216 feature columns in the documented workflow. PCA reduced the representation to 500 dimensions.
Documented shape: (696, 500)
Data type: float32
Reported explained variance: approximately 95.43%
Artifact: real_expression_500d.npz
Recorded keys: expression, sample_ids
The 95.43% figure refers to variance retained by the selected PCA components in the data used to fit that transformation. It is not model accuracy, clinical accuracy, or the percentage of biology preserved.
Mutation
The source mutation matrix had approximately 19,506 feature columns in the documented workflow. PCA reduced the representation to 64 dimensions.
Documented shape: (696, 64)
Data type: float32
Reported explained variance: approximately 56%
Artifact: real_mutation_64d.npz
Recorded keys: mutation, sample_ids
PCA components are weighted combinations of original variables. A 64-dimensional mutation PCA vector is not equivalent to 64 individual named mutations.
PCA leakage prevention
For a strict held-out evaluation, fit PCA on the training partition only. Apply the fitted transformation to validation and test partitions without refitting. The same rule applies to data-dependent scaling, imputation, feature selection, and other preprocessing.
If PCA is fitted using all cell lines before the split, the held-out feature distribution has influenced preprocessing. This is a form of information leakage even if response labels were not used to fit PCA.
Identifier alignment
Maintain explicit mappings between:
Source cell-line identifiers.
Canonical cell-line identifiers used in the integrated cohort.
Response-table cell-line IDs.
Drug identifiers.
Chemical structure identifiers and SMILES.
Rows of each processed feature matrix.
Every training example must point to the correct cell-line features, drug features, and measured response. Save mappings and audit unmatched/duplicate IDs rather than silently dropping or reassigning them.
Missing data and unresolved mappings
Do not invent drug structures, molecular measurements, biology features, or response labels. Record missingness, exclusion rules, and the number of examples affected by each filter.
The missing 64-dimensional biology feature artifact is a known dependency in the documented workflow. A valid biology representation must be defined, generated, versioned, and aligned before it can be used in a real-data inference or training run.
9. Training, validation, and evaluation
Training is distinct from inference
A forward pass computes model outputs. Training additionally requires target labels, a loss function, gradient computation, an optimizer, and repeated parameter updates. A valid scientific training run also requires an appropriate dataset split and a reproducible experiment configuration.
Regression formulation
For a response target such as log2 AUC, a regression model may be written as:
[ \hat{y}{=tex}i=f\theta{=tex}(x_i) ]
A common loss is mean squared error:
[ \mathrm{MSE}{=tex}=\frac{1}{N}{=tex}\sum{=tex}_{i=1}^{N}(y_i-\hat{y}{=tex}_i)^2 ]
This is a general example, not a claim that every CANXURE training path uses MSE on real GDSC data. Inspect the exact script and checkpoint metadata for the loss used in any reported experiment.
Metrics
MAE: average absolute prediction error in the target's units.
RMSE: square root of mean squared error; penalizes large errors more heavily.
R²: compares the model's squared error with the variance around the target mean; negative R² means the model performs worse than a mean-prediction baseline under the standard definition.
Correlation or ranking metrics: can provide complementary information, but must be selected before analysis and reported with appropriate uncertainty.
Concordance index (C-index): a survival/ranking metric in relevant time-to-event settings; it should not be used to imply a Cox survival model when the target is GDSC AUC.
Initial documented held-out baseline
The project's earlier real-data work recorded the following evaluation snapshot:
Metric            Reported value
RMSE                      0.5662 MAE                       0.4061 R²                       -0.4776 Test examples             43,402
These values are historical, preliminary results from the documented work and should be reproduced from the exact code, split, preprocessing artifacts, checkpoint, and target definition before publication or use in a presentation.
The negative R² indicates that the evaluated model did not outperform the standard test-mean baseline on squared error. It is an important limitation, not evidence of strong predictive performance.
Documented cell-line split
The earlier evaluation notes recorded:
Training cell lines: 487
Validation cell lines: 104
Test cell lines: 105
Verify that this split was applied consistently to all response rows and that preprocessing was fitted only on training data before describing the experiment as leakage-free.
Leakage and generalization
Randomly splitting cell-line--drug pairs can place the same cell line in both training and testing. This may be acceptable for some interpolation objectives, but it does not test generalization to unseen cell lines.
Choose the split based on the intended research question:
Interaction-level split: predicts held-out combinations when cell lines and drugs may already be seen.
Cell-line-held-out split: evaluates generalization to unseen cell lines.
Drug-held-out split: evaluates generalization to unseen drugs.
Both held out: tests a more demanding setting involving unseen cell lines and unseen drugs.
Report the split policy and all preprocessing decisions. Compare with simple baselines, including a mean predictor and appropriate classical models. Where possible, report confidence intervals or variability across repeated splits.
Survival analysis is a different task
A Cox proportional-hazards objective requires time-to-event information and event/censoring indicators. GDSC log2 AUC is not, by itself, survival data. Do not describe the current response task as DeepSurv/Cox modelling unless the target data and implementation truly use a survival objective.
10. Repository layout
The repository is organized as a modular research framework. The following is a high-level guide to the main areas; individual files may change as development continues.
Canxure/
├── app/                 # Optional user-facing application components
├── config/              # Configuration files and experiment settings
├── data/                # Data directory structure/placeholders; large datasets may be external
├── docs/                # Project and technical documentation
├── models/              # Model artifacts/checkpoint organization
├── notebooks/           # Exploratory and orchestrating notebooks
├── outputs/              # Generated reports, predictions, summaries, and metrics
├── scripts/              # Command-line entry points for demo/training workflows
├── src/
│   ├── biology/          # Biology and tumor-profile abstractions
│   ├── chemistry/        # Chemical representations/fingerprints where implemented
│   ├── data/             # Demo data and data handling
│   ├── models/           # Research model and model metrics
│   ├── reporting/        # Report generation
│   ├── simulation/       # Virtual simulation components
│   ├── training/         # Checkpointing and training utilities
│   └── utils/             # Reproducibility/environment helpers
├── tests/                # Automated tests
├── requirements.txt      # Runtime/general dependencies
├── requirements-dev.txt  # Development and test dependencies
├── pyproject.toml        # Project metadata and tool configuration
└── README.md
This tree is an orientation map rather than an exhaustive manifest. Use the current repository tree as the source of truth for the exact files present in a particular commit.
Main entry points
Path                                Purpose
src/models/research_model.py      Defines ResearchDigitalTwin
src/models/metrics.py             Parameter counts, memory estimates, checkpoint-size and inference benchmarking helpers
scripts/run_simulation.py         Runs the synthetic demonstration workflow with --demo
scripts/train_research.py         Executes a minimal research-model training/smoke-test step with random tensors
requirements.txt                  General Python dependencies
requirements-dev.txt              General dependencies plus testing/linting/type-checking tools
pyproject.toml                    Project metadata and pytest/Ruff configuration
11. Environment and installation
Requirements
Python >=3.10
Git
pip
CPU-capable PyTorch for the neural-model scripts
Optional CUDA-compatible PyTorch and NVIDIA GPU for accelerated training
Packages listed in requirements.txt
Development tools from requirements-dev.txt if running tests and lint checks
RDKit and PyTorch Geometric can require environment-specific installation choices. The currently documented fingerprint-based model does not require a GNN merely to accept a precomputed 1,024-dimensional drug vector. Install only the dependencies needed for the workflow you are running.
Clone the repository
git clone https://github.com/battlehubindia81-glitch/Canxure.git
cd Canxure
Create a virtual environment
Linux/macOS:
python -m venv .venv
source .venv/bin/activate
Windows PowerShell:
python -m venv .venv
.venv\Scripts\Activate.ps1
Install development dependencies
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
If you only need the general dependencies:
pip install -r requirements.txt
The repository's dependency files may not install every optional package required by a separate real-data notebook. Follow the notebook and feature-generation script requirements for RDKit, PyTorch, and any optional graph-learning packages.
Check the Python and PyTorch environment
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
To inspect the repository's environment report helper, use the relevant scripts or inspect src/utils/environment.py.
GPU note
The training script supports --device auto, --device cpu, and --device cuda. With auto, it selects CUDA if available and otherwise CPU. Explicitly requesting CUDA on a runtime without available CUDA raises an error.
PyTorch, CUDA, GPU drivers, and optional PyTorch Geometric versions must be mutually compatible. Do not install a random PyG wheel without matching the active PyTorch/CUDA environment.
12. Running the demonstration workflow
The demo is the recommended first check after installation.
python scripts/run_simulation.py --demo
The script generates synthetic expression data, preprocesses it, constructs a demonstration tumor profile, runs the virtual simulation, and writes outputs.
Expected outputs
The script is documented to write:
outputs/
├── predictions/
│   └── demo_predictions.csv
├── reports/
│   └── digital_twin_report.md
└── model/
    ├── model_summary.json
    └── model_summary.txt
The script also prints a summary of the synthetic demonstration and includes a research-use-only warning.
How to interpret demo output
Demo samples are generated for software demonstration.
Demo results are not trained/validated evidence from the real DepMap/GDSC pipeline.
A ranked candidate in a demo report is not a treatment recommendation.
Do not use the demo's scores as estimates of real patient efficacy or toxicity.
Re-run the script after a clean clone to confirm that the current environment and output directories work.
13. Running the research-model smoke-training script
The script below builds the ResearchDigitalTwin, performs a minimal optimization step, saves a checkpoint, and writes a model summary.
python scripts/train_research.py --device auto
Force CPU:
python scripts/train_research.py --device cpu
Request CUDA when available:
python scripts/train_research.py --device cuda
Resume from a compatible checkpoint:
python scripts/train_research.py --device auto --resume models/research/checkpoints/latest.pt
The checkpoint path and resume compatibility depend on the artifact present in the working tree.
Critical interpretation note
The current script creates random tensors for expression, mutation, biology, drug, and the target. It performs a single optimization step as a smoke test. It is useful for testing that:
The model can be instantiated.
The forward pass accepts expected shapes.
A loss can be calculated.
Gradients can be backpropagated.
An optimizer update can run.
A checkpoint and model summary can be written.
It is not a full training pipeline on the 255,133 real cell-line--drug pairs. Its reported loss, model summary, or checkpoint must not be presented as scientific validation of CANXURE's real-data predictive ability.
Expected model artifacts
The script writes:
models/research/checkpoints/latest.pt
outputs/model/model_summary.json
outputs/model/model_summary.txt
The summary includes parameter counts, per-module parameter counts, memory estimates, checkpoint size, a small inference benchmark, training-step timing, loss, device, and environment information. Hardware-dependent timings should not be compared without matching conditions.
14. Reproducing the real-data pipeline
The real-data work has been orchestrated in Google Colab, with files retained in Google Drive to survive runtime resets. A documented notebook name is Canxure_Real_Data_Pipeline.ipynb, and the historical backup root is:
/content/drive/MyDrive/Canxure_backup
The precise notebook and artifact versions should be checked before executing a reproduction. This README does not claim that all raw datasets, derived artifacts, or model checkpoints are bundled in GitHub.
Recommended high-level sequence
Prepare the environment. Clone/pull the intended repository revision and install compatible dependencies.
Acquire source data. Obtain the relevant DepMap and GDSC releases under their applicable terms. Record versions and checksums.
Audit source schemas. Inspect sample identifiers, feature columns, missing values, duplicate measurements, drug IDs, and target definitions.
Define a common cell-line cohort. Use documented identifier mappings and audit the final intersection.
Build the response table. Preserve the measured response target and document duplicate consolidation and filtering.
Prepare molecular matrices. Align expression and mutation rows to the common cell-line IDs.
Split before fitting data-dependent transformations. For a strict cell-line holdout, partition cell lines first.
Fit preprocessing on training data. Fit PCA and other learned preprocessing only on the training partition; transform validation/test with the fitted objects.
Map drug structures. Map GDSC drug IDs using a trustworthy resource; validate structures with RDKit; preserve unresolved mappings as unresolved.
Generate drug fingerprints. Save the fingerprint algorithm, parameters, drug ID order, and matrix shape alongside the feature file.
Resolve biology features. Generate a documented 64-dimensional biology representation or update the architecture/experiment configuration explicitly. Do not fabricate features to satisfy tensor dimensions.
Construct aligned examples. Verify that each row's cell-line features, drug features, and response target refer to the same pair.
Train on real targets. Use a documented training loop and loss suitable for the target, with validation and checkpoint selection.
Evaluate on held-out data. Report metrics, baseline comparisons, split strategy, coverage, and limitations.
Save a complete experiment record. Include configuration, random seeds, data version, mapping files, preprocessing objects, checkpoint, metric outputs, and environment details.
Artifact format notes
The documented expression and mutation NPZ artifacts used allow_pickle=True when loaded in the earlier workflow and recorded sample_ids alongside the arrays. NPZ files can contain multiple named arrays. Inspect their keys and array shapes before loading them into a model.
Example inspection snippet:
import numpy as np

path = "data/processed/real_expression_500d.npz"
with np.load(path, allow_pickle=True) as data:
    print("Keys:", data.files)
    for key in data.files:
        arr = data[key]
        print(key, arr.shape, arr.dtype)
Only load NPZ files from trusted sources. allow_pickle=True may permit loading Python objects and should not be used casually with untrusted files. If object arrays are not required, prefer allow_pickle=False and store identifiers using safe string/numeric dtypes.
What a reproducible real-data run must preserve
At minimum, record:
Repository commit SHA.
Python, PyTorch, NumPy, pandas, scikit-learn, and RDKit versions.
Source dataset names/releases and access dates.
Input file hashes or other integrity identifiers.
Cell-line and drug ID mapping versions.
Inclusion/exclusion rules and final counts at each stage.
PCA fitted objects and explained-variance values.
Fingerprint type, size, parameters, and drug ID order.
Biology-feature definition and artifact version.
Train/validation/test IDs and split seed.
Training configuration, optimizer, learning rate, epochs, loss, and checkpoint-selection rule.
Model checkpoint hash.
Evaluation script, metrics, target scale, and baseline results.
15. Artifacts and expected outputs
Outputs are generated artifacts, not source-of-truth data. Keep their provenance clear.
Artifact                Purpose                    Important metadata
Raw source tables       Original downloaded data   Source, release, download date, checksum
Consolidated response   Cell-line--drug response   Target definition, table                   target                     filtering, duplicate handling
Expression NPZ          Reduced expression         PCA fit source, features and sample IDs    component count, sample order
Mutation NPZ            Reduced mutation features  PCA fit source, and sample IDs             component count, sample order
Drug fingerprint NPZ    Fingerprints and drug IDs  RDKit version, algorithm, parameters, mapping source
Model checkpoint        Learned parameter state    Architecture, training config, data version, split, seed
Model summary           Parameter/memory/runtime   Device, batch size, measurements               software/hardware versions
Prediction CSV          Per-example outputs        Checkpoint, input IDs, target scale, run identifier
Research report         Summary of a simulation or Data provenance, experiment                 limitations, intended interpretation
Do not commit large datasets, credentials, patient data, or unlicensed source files to the repository. Use appropriate storage and access controls for external artifacts.
16. Reproducibility and data integrity
Random seeds
Set and record random seeds for Python, NumPy, and PyTorch where applicable. Some GPU operations may remain nondeterministic depending on hardware and kernel configuration. A seed alone does not guarantee bit-for-bit reproducibility.
Stable identifiers
Use stable IDs rather than row positions to align data. Save the ordered ID arrays used to create each feature matrix. Do not assume two files share row order because they have the same number of rows.
Duplicate consolidation
The documented response workflow averaged pairs with two measurements in one consolidation step. Preserve the aggregation rule and record counts before and after consolidation. Averaging repeated measurements may reduce duplicate rows, but it does not remove every possible source of experimental dependence.
Split integrity
For a cell-line-held-out evaluation, verify that no held-out cell-line identifier appears in training or validation. For a drug-held-out evaluation, verify the analogous condition for drug IDs. Audit aliases and duplicate identifiers.
Baselines
A complex neural model should be compared with transparent baselines. Depending on the task, these may include a mean predictor, regularized linear regression, random forest, gradient boosting, or a simpler single-modality model. All methods should use the same data splits and target definition.
Uncertainty and calibration
Report variability across splits or seeds when feasible. If an output is described as a probability or risk score, evaluate calibration and define the event/label that probability refers to. A sigmoid or softmax activation alone does not establish calibration.
17. Current limitations and known open research tasks
The items below reflect the documented state of the project and should be re-checked against the latest code and experiment artifacts before each release.
17.1 Preliminary model performance
The recorded held-out evaluation had negative R². This indicates that the evaluated model did not outperform the test-mean baseline on squared error. More work is required to establish whether the chosen representations and architecture improve predictive performance.
17.2 Biology feature dependency
The architecture expects 64 biology features, but the previously documented real-data work did not locate the corresponding feature artifact. A valid, reproducible biology feature definition and generation process is needed before claiming a fully reproducible real-data model run with all four modalities.
17.3 Incomplete drug structure coverage
The documented structure mapping covered 421 of 537 drugs. The remaining 116 were not assigned invented structures. More trustworthy mapping or an explicitly justified missing-structure strategy is needed to increase coverage.
17.4 Different training paths
The repository smoke-training script uses random tensors; it is not the real-data training pipeline. The real-data workflow must have its own explicit training entry point, configuration, and reproducible outputs before it can be reproduced from a fresh clone without manual notebook state.
17.5 Output heads need target-level documentation
The model defines efficacy, toxicity, resistance, and clinical-response heads. Each head requires an explicit target definition, label provenance, loss, sample coverage, evaluation metrics, and interpretation guidance. Output names alone do not establish that these endpoints were trained or validated.
17.6 Cell-line models do not represent complete tumors
In-vitro cell-line datasets generally do not reproduce the full tumor microenvironment, immune system, stromal interactions, tissue architecture, or patient-specific physiology.
17.7 Pharmacology is incomplete
A drug fingerprint does not encode the complete effects of dose, absorption, distribution, metabolism, excretion, exposure duration, target engagement, or drug--drug interactions. Cell-line response is not equivalent to clinical efficacy or safety.
17.8 Generalization must be tested explicitly
Performance on familiar cell lines or drugs may not transfer to unseen biological systems or unseen compounds. The intended deployment/generalization setting must determine the split strategy and evaluation design.
17.9 Interpretability is not causality
Feature importance, attention weights, correlations, or model explanations can generate hypotheses. They do not independently prove that a gene or mutation causes sensitivity or resistance. Mechanistic claims require appropriate experimental and scientific evidence.
18. Responsible interpretation
CANXURE is intended for research exploration.
Do not use outputs to prescribe or recommend treatment.
Do not infer patient benefit from cell-line AUC alone.
Do not treat model-generated toxicity scores as validated clinical safety probabilities.
Do not claim a patient-specific digital twin without the data, validation, and dynamic modelling needed to justify that description.
Do not present synthetic demo outputs as real experimental results.
Do not claim a GNN, survival model, or a specific loss function unless the source and experiment actually implement it.
Report failed experiments and negative results.
Keep source data provenance and all preprocessing decisions auditable.
Follow dataset terms, institutional requirements, and applicable privacy rules.
A responsible description of the current project is:
CANXURE is a digital-twin-inspired computational research prototype exploring multimodal integration of cancer-cell molecular features and drug representations to study experimentally measured drug-response patterns. Its predictive performance and biological relevance require further rigorous validation.
19. Troubleshooting
ModuleNotFoundError when importing src...
Run commands from the repository root, where pyproject.toml and src/ are located. Confirm that the active environment has the required dependencies installed.
pwd
python -c "import sys; print(sys.executable)"
CUDA requested but unavailable
Use:
python scripts/train_research.py --device auto
or explicitly use CPU:
python scripts/train_research.py --device cpu
Check that the installed PyTorch build supports the CUDA environment before requesting --device cuda.
Input dimension mismatch
Check each input shape:
print(expression.shape)  # expected (B, 500)
print(mutation.shape)    # expected (B, 64)
print(biology.shape)     # expected (B, 64)
print(drug.shape)        # expected (B, 1024)
Also verify that all rows are aligned to the same examples. Do not solve a shape error by padding or inventing features without a scientifically defined transformation.
Missing biology feature file
The documented model expects 64 biology features. Locate the exact feature-generation script and source artifact, or document the dependency as unresolved. Do not replace it with arbitrary random features for scientific inference.
RDKit installation issues
RDKit installation can depend on Python version and operating system. Use an environment compatible with the RDKit release required by the feature-generation workflow. If only precomputed fingerprints are needed for a model forward pass, RDKit may not be required at inference time, provided the fingerprint artifact was generated correctly and its provenance is available.
Different results after rerunning
Check seeds, source versions, split IDs, fitted preprocessing, checkpoint selection, library versions, GPU/CPU behavior, and deterministic settings. Confirm that the evaluated checkpoint is the one you intended to evaluate.
No real data after cloning
The raw DepMap/GDSC data and Drive-backed artifacts may not be committed. Follow the real-data workflow, acquire authorized source datasets, and regenerate artifacts. Do not interpret a successful demo run as evidence that the external datasets were loaded.
20. Roadmap
Potential research and engineering priorities include:
Reproducible data pipeline: convert notebook-only steps into versioned scripts with explicit inputs and outputs.
Data provenance: record dataset releases, hashes, mapping tables, and filtering counts.
Biology features: define and validate a defensible biology representation compatible with the 64-dimensional model input, or revise the architecture transparently.
Drug coverage: improve trustworthy compound mapping for unresolved GDSC drug IDs.
Leakage-resistant evaluation: fit preprocessing within each training split and hold out cell lines and/or drugs according to the scientific question.
Baseline comparisons: evaluate simple baselines and single-modality variants alongside multimodal models.
Target-specific training: document and train each output head only where valid target labels exist.
Uncertainty and calibration: evaluate prediction uncertainty, calibration, and robustness.
Interpretability: use carefully validated analyses to generate hypotheses, followed by independent biological evidence.
Architecture comparisons: evaluate fingerprint-based MLPs against graph-based drug encoders only when both are implemented and tested fairly.
Independent validation: assess transfer to separate datasets and clearly define the limits of external validity.
Documentation and testing: automate shape checks, ID-alignment tests, split-integrity checks, and end-to-end reproduction tests.
This roadmap is aspirational. It does not imply that the listed work is already complete.
21. Contributing and research collaboration
Contributions that improve correctness, reproducibility, or scientific transparency are especially valuable.
Useful contribution areas
Bug reports with a minimal reproducible example.
Data-schema and identifier-mapping audits.
Tests for input shapes and row alignment.
Documentation of target construction and data provenance.
Leakage-resistant split and evaluation utilities.
Baseline models and fair comparison protocols.
Feature-generation scripts with explicit versioning.
Model interpretability analyses with appropriate caveats.
Reproducible environment and installation improvements.
Suggested experiment record
For each experiment, record:
experiment_name: example_experiment
repository_commit: "<git-sha>"
dataset_release: "<source-release>"
response_target: "GDSC log2 AUC"
split_strategy: "cell-line-held-out"
random_seed: 42
expression_components: 500
mutation_components: 64
drug_fingerprint_dimensions: 1024
biology_feature_version: "<version-or-unresolved>"
model_checkpoint: "<path-or-artifact-id>"
loss: "<exact-loss-used>"
metrics:
  rmse: null
  mae: null
  r2: null
notes: "<limitations and observations>"
This is a template. Replace placeholders with actual values; do not leave a placeholder in a published result.
Development checks
pytest
The repository's development dependencies include ruff and mypy; use them where applicable:
ruff check .
mypy src
A passing test suite establishes only that the covered tests pass. It does not validate scientific claims or clinical usefulness.
22. Data sources and acknowledgements
The documented research direction draws on:
DepMap, for cancer-cell molecular profiles.
Genomics of Drug Sensitivity in Cancer (GDSC), for experimental drug-response measurements.
RDKit, for cheminformatics and molecular fingerprint generation.
PyTorch, for neural-network implementation and optimization.
NumPy, pandas, SciPy, and scikit-learn, for numerical processing and data science utilities.
Users must cite the relevant dataset releases and source publications required by each data provider. Consult the official data portals and terms for current access conditions and citation instructions:
DepMap: https://depmap.org/portal/
GDSC: https://www.cancerrxgene.org/
RDKit: https://www.rdkit.org/
PyTorch: https://pytorch.org/
A drug structure mapping file from a third-party research resource was used in the documented workflow. Cite the exact resource and version in any work that relies on it; do not treat a mapping source as authoritative without checking its provenance and coverage.
23. License and citation
License
Check the repository's current LICENSE file before redistributing code or artifacts. If no license is present, do not assume that the code is licensed for unrestricted reuse. Dataset and third-party resource licenses are separate from the repository's code license.
Citation
If you use CANXURE in research, cite the repository revision, access date, and the exact datasets and software versions used. Until a formal paper or DOI is available, a repository citation can follow this general format:
Singh, N. CANXURE AI: Precision Oncology Virtual Tumor Research Framework. GitHub repository, battlehubindia81-glitch/Canxure. Include the commit SHA and access date for reproducibility.
Do not cite preliminary metrics as published benchmark results unless the experiment has been independently reproduced and the protocol is fully documented.
Final note
CANXURE is a work in progress at the intersection of healthcare, computational biology, and machine learning. The most important measure of progress is not how advanced the architecture sounds, but whether the data are trustworthy, the experiment is reproducible, the evaluation is rigorous, and the conclusions stay within the evidence.
Research use only --- not a clinical treatment recommendation.