import yaml


def yaml_to_env(config_file: str) -> str:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    env_config = ''

    for param, value in config.items():
        if isinstance(value, dict):
            final_param = param

            while isinstance(value.values(), dict):
                final_param += f'.{value.values()}'

                value = value.values()

            final_param += f'{value.values()}'
            env_config += final_param + '\n'

        else:
            env_config += f'{param}={value}' + '\n'

    print(env_config)


def env_to_yaml(env_list: str) -> str:
    ...


if __name__ == '__main__':
    yaml_to_env('YAML_CONFIG/config.yml')