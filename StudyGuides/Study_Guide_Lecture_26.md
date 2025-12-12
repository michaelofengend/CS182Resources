Alright, let's get you ready to ace this section on post-training, test-time compute, and diffusion models. This is a hefty topic, but we'll break it down into digestible pieces. Get ready for some deep dives!

### Thorough Study Guide: Post-Training, Test-Time Compute & Diffusion Models

#### I. Core Concepts

1.  **Post-Training & Instruction Following:**

    *   **Intuition**:  We start with a pre-trained Language Model (LLM), but it's like a talented but undisciplined student.  It knows a lot, but doesn't know *how* to apply it to specific tasks.  Instruction Following is the process of teaching the model to follow instructions well.  We're basically turning it into a super-obedient and helpful assistant.

    *   **Mechanism:**  This is typically done with Supervised Fine-Tuning (SFT). The key is to feed the model a dataset of *instructions* paired with *desired outputs*. The model then learns to maximize the probability of generating the correct output given an instruction.

    *   **Analogy**: Think of it like training a dog.  The pre-trained model is the dog knowing all the words, SFT is teaching it specific commands (sit, stay, fetch) and rewarding the correct actions.
        *   The lecture notes show SFT as maximizing cross-entropy loss over the "first response token" and "second response token". This is the training signal to align the LLM with our instructions, while masking the loss on the question.

2.  **Improving Model Performance: Compute vs. Training:**

    *   **Intuition**: Once we can follow instructions, how do we make the model *better* at solving problems? There are two broad approaches: (A) Spend more effort *while* answering, and (B) *retrain* the model to be better.

    *   **(A) Test-Time Compute (Inference):**
        *   **Intuition:** Like a student who can take their time on a test, we allow the model to do more "thinking" at inference time. The oldest, and simplest way is pure prompting with careful instructions. Another method is chain-of-thought prompting, where you ask the model to show its work. More advanced techniques involve sampling responses and repeated generation with voting or grading models.
        *   **Mechanism**:
            *   *Pure Prompting*: Careful prompt engineering to guide the model's reasoning.
            *   *Repeated Generation*: Generate N possible answers, then use a majority vote, or train a reward model to grade the answers, and pick the best one.
        *   **Analogy**:  Imagine asking multiple experts to solve the same problem, then combining their answers or having another expert judge the best solution. Repeated generation is like asking multiple experts, and using a reward model is like having an experienced lead expert judge the responses.

    *   **(B) Post-Training (Retraining):**
        *   **Intuition**: After the model is pre-trained and aligned to instructions, we can further train it to improve its problem-solving skills by making it more willing to spend compute.
        *   **Mechanism**: One of the methods to do this is Reinforcement Learning with Verifiable Rewards (RLVR).
            *   With RLVR, we need a verifiable reward signal, where the agent gets rewarded based on an autograder.
        *   **Analogy:** Think of it like giving the student practice exams after school and grading them on whether they arrive at the correct answers. With RLVR, you have an automated process that gives the agent rewards to solve problems.

3.  **Reinforcement Learning with Verifiable Rewards (RLVR):**

    *   **Intuition**:  We want the model to be better at tasks where we have a clear way to *verify* the answer.  This is crucial for math, coding, or any problem where we have an "autograder".

    *   **Mechanism**:

        *   Generate multiple solutions to the same problem (G generations).
        *   Use an autograder to score each solution.
        *   Use a REINFORCE trick to update the LLM such that better generations become more probable. This is essentially a Reparameterization Gradient Estimator, using verifiable rewards as a guide.

        *   **Important Note:** The gradient from the autograder must **not** flow back into the autograder.

    *   **Analogy**:  It's like training a self-driving car. The car tries different routes (generations). The GPS system (autograder) verifies if it arrived at the destination. The car then learns to favor routes that led to success.

    *   **Key Idea**: The RLVR can be understood from a "Bas of tokens perspective". For the generation loss, every token inherits its rewards from its generation.

    *   **Challenge**: It is impractical to conduct backpropagation on the LLM because of the amount of compute the agent is using (multiple generations). To amend this, RLVR is used with Important Sampling to correct any discrepancies.

    *   **Important Sampling**: In RLVR, a correction must be made due to generating with a different probability than we are training with. Important Sampling applies a ratio correction. Additionally, a stop gradient is needed so that the adjustment doesn't affect the gradient.

    *   **Capping**: RLVR may overweight unlikely samples. To avoid this, RLVR uses a capping mechanism, capping the reward and avoiding overweighting.

4.  **RLHF (Reinforcement Learning with Human Feedback):**

    *   **Intuition:** We want the model to align with complex human values and preferences that can't be easily autograded (e.g., helpfulness, harmlessness, creativity). We need to use *human* feedback as the reward signal.

    *   **Mechanism**:

        *   Train a *reward model* that predicts human preferences.

        *   Generate different outputs and ask humans to rank or compare them.

        *   Train the reward model to predict these human preferences.

        *   Use this reward model in a standard RL algorithm (e.g., REINFORCE) to optimize the language model.

    *   **Analogy**:  Think of training a comedian.  You can't autograde humor. You need a live audience (humans) to laugh or boo. You use that feedback to shape the comedian's performance.

5. **DPO (Direct Preference Optimization):**

    *   **Intuition**: To simplify RLHF, can we bypass the step of training an explicit reward model? Turns out, yes! We can directly optimize the LLM using human preference data.
    *   **Mechanism**: DPO formulates the problem such that it only requires the LLM parameters. The DPO loss function directly compares the LLM's outputs to human preference data. DPO is different from RLVR in that it doesn't require autograders, as humans are directly grading the quality.
    *   **Analogy**: Imagine trying to teach someone to cook. Instead of first training a "taste model" to predict what's delicious, you just show the person pairs of dishes and tell them which one is better. They then directly adjust their cooking techniques to match those preferences.
    *   **Benefits**: Simplifies the RLHF pipeline, more stable training.
        *   The title of the DPO paper is "Your Language Model is Secretly a Reward Model". What does that mean? It means that high probability generations can be deemed "good" and low probability generations can be deemed "bad".

6.  **TRDPO (Trust Region Direct Preference Optimization):**

    *   **Intuition:** Improve DPO by using an updated reference policy at fixed intervals.
    *   **Mechanism**: A new reference policy is used to help guide the agent's behavior.
    *   **Analogy**: Periodically adjusting the reference to a new baseline to help the agent.

7.  **Diffusion Models:**

    *   **Intuition**:  Instead of directly generating an image, can we learn to "gradually refine" random noise into an image?  Diffusion models are inspired by non-equilibrium thermodynamics in physics.

    *   **Forward Diffusion Process**:  Gradually add noise to an image until it becomes pure noise (Gaussian noise). This is a Markov process (the future depends only on the present state, not the past).

    *   **Reverse Diffusion Process**:  Learn to *reverse* this process, starting from pure noise and *iteratively denoising* to create an image.

    *   **Analogy**: Imagine starting with a blurred image and gradually sharpening it, or starting with white noise, and gradually adding strokes to form a recognizable picture.

    *   **Key Design Question**: Are the backward paths deterministic, or stochastic?
        *   DDPM is stochastic, and depends on randomness
        *   DDIM is deterministic, and can be calculated without randomness.
    *   **Why "Diffusion?"**:  Think of a drop of dye spreading out in water until it's evenly distributed. That's equilibrium.  We learn to reverse this process, starting from the equilibrium state (random noise) and creating a non-equilibrium state (a coherent image).

    * It helps to understand the forward and reverse process for diffusion models through a descrete approach. The course's discussion covers this via zeroes and ones.

#### II. Key Analogies

*   **Instruction Following:**  Training a dog (SFT is like teaching specific commands).
*   **Repeated Generation:**  Multiple experts solving the same problem.
*   **Reward Model:** An expert judge.
*   **RLVR:**  Training a self-driving car with GPS.
*   **DPO:** Teaching someone to cook by showing them pairs of dishes and which one is better.
*   **Diffusion Models:** Sharpening a blurred image.
*   **Diffusion Process:** Dye spreading in water.

#### III. Math Decoded

*   **Cross-Entropy Loss (SFT):** Measures the difference between the model's predicted probability distribution and the desired distribution (one-hot vector representing the correct token).

*   **REINFORCE Trick (RLVR):**  A way to get gradients through a non-differentiable reward function.
    *   `∇E[F(y)] = E[F(y) ∇log p(y)]`. We use this in Homework 3.  It means we can optimize the *expectation* of a reward, even if the reward itself isn't differentiable.  We achieve this by weighting the gradient of the log probability with the reward. This turns the original function non-differentiable, but makes the expectation differentiable.
*   **Kullback-Leibler Divergence (KL Divergence):** A measure of how different two probability distributions are. Used as a regularizer to keep the LLM near the intended distributions.

#### IV. Practice Insights

*   Discussion worksheet will ask you to make connections between DDPM models and the descrete case. You'll have to calculate probability distributions on the reverse path, and other mathematical operations to show a thorough understanding.

*   The professor says that knowing the relationships between different tokens and parameters can be helpful during an exam.

*   Consider what methods to train the reward model when solving tasks, and avoid using reward models when they aren't applicable to the task at hand.

*   For a good model with SFT, the parameters for the questions and the answers should differ.

#### General Tips

*   **Focus on the "Why":** Always ask yourself *why* a particular technique is used. What problem is it trying to solve? What are its limitations?

*   **Connect the Dots:**  Try to relate new concepts to things you already understand. This makes learning more efficient and helps you retain information better.

*   **Practice, Practice, Practice:** Work through the exercises in the discussion worksheets and homework assignments. This is the best way to solidify your understanding.

*   **Don't Be Afraid to Ask for Help**: Neural networks can be complex. Use the discussion forum on ED and the TAs to help get you to success.