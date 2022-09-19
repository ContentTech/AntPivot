import json
import os
import pickle
import yaml
import torch


def load_yaml(filename):
    with open(filename, "r") as yaml_file:
        return yaml.load(yaml_file)


def load_json(filename):
    with open(filename, "r") as json_file:
        return json.load(json_file)


def save_json(content, filename):
    with open(filename, "w") as json_file:
        json.dump(content, json_file)


def create_checkpoint_dir():
    if not os.path.exists("saved"):
        os.mkdir("saved")


def save_model(model, checkpoint_name, other_states=None):
    create_checkpoint_dir()
    print('Model saved as: {}'.format(checkpoint_name))
    saved_dict = {
        "model_parameter": model.state_dict()
    }
    if other_states is not None:
        saved_dict.update(other_states)
    torch.save(saved_dict, "saved/" + checkpoint_name)


def load_model(model, checkpoint_name):
    print('Model loaded from {}'.format(checkpoint_name))
    loaded_dict = torch.load("saved/" + checkpoint_name)
    model.load_state_dict(loaded_dict["model_parameter"])
    return loaded_dict


def create_file(path):
    if not os.path.exists(path):
        f = open(path, 'w')
        f.close()


def save_pickle(obj, filename):
    with open(filename, "wb") as pickle_file:
        pickle.dump(obj, pickle_file)


def load_pickle(filename):
    with open(filename, "rb") as pickle_file:
        return pickle.load(pickle_file)


if __name__ == "__main__":
    print(load_yaml("../config/nad.yaml"))
