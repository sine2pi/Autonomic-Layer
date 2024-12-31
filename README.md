Dynamic Auto Adjustments in Transformers

The AutonomicLayer class is kind of like a central hub for dynamic auto adjustments of hyperparameters in my transformer model, featuring an in-progress meta-learner. Currently, it adjusts the base frequency (rope) and the window size of the hybrid attention block based on training loss.

Note: At the moment, the adjustment mechanism is quite aggressive (set to an extreme level).

Key Features
Dynamic Feedback: Automatically adjusts the window size of the hybrid attention block and the base frequency of the given's rotary embeddings based on the model's performance.

Hybrid Attention Block: Uses multi-head attention blocks for both small and large scales. The infrastructure for more complex adjustments is in place, but further experimentation is needed.

Work in Progress: Some of the blocks are not yet connected in this version of the model. However, the dynamic feedback mechanism is fully functional.

Hugging Face Integration: Includes a full script with Hugging Face trainer and dataset integration for ease of use. This was added for compatibility purposes, though the model performs better with a PyTorch loop.
