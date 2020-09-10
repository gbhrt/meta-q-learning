Meta-Q-Learning
=============================================
This paper introduces Meta-Q-Learning (MQL), a new off-policy algorithm for meta-Reinforcement Learning (meta-RL). MQL builds upon three simple ideas. First, we show that Q-learning is competitive with state-of-the-art meta-RL algorithms if given access to a context variable that is a representation of the past trajectory. Second, a multi-task objective to maximize the average reward across the training tasks is an effective method to meta-train RL policies. Third, past data from the meta-training replay buffer can be recycled to adapt the policy on a new task using off-policy updates. MQL draws upon ideas in propensity estimation to do so and thereby amplifies the amount of available data for adaptation. Experiments on standard continuous-control benchmarks suggest that MQL compares favorably with the state of the art in meta-RL.

This repository provides the implementation of [Meta-Q-learning](https://arxiv.org/abs/1910.00125). If you use this code please cite the paper using the bibtex reference below.

```
@misc{fakoor2019metaqlearning,
    title={Meta-Q-Learning},
    author={Rasool Fakoor and Pratik Chaudhari and Stefano Soatto and Alexander J. Smola},
    year={2019},
    eprint={1910.00125},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
Be sure to:

* Change the title in this README
* Edit your repository description on GitHub
* Write in your license below and create a LICENSE file

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This code is licensed under the CC-BY-NC-4.0 License.

# Contact

Please open an issue on [issues tracker](https://github.com/amazon-research/meta-q-learning/issues) to report problems or to ask questions or send an email to me, [Rasool Fakoor](https://github.com/rasoolfa).
