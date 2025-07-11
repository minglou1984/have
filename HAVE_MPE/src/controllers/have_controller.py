#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch as th
import copy

from .basic_controller import BasicMAC


class DataParallelAgent(th.nn.DataParallel):
    def init_hidden(self):
        # make hidden states on same device as model
        return self.module.init_hidden()


# This multi-agent controller shares parameters between agents
class HAVEMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(HAVEMAC, self).__init__(scheme, groups, args)
        self.n_enemies = args.n_enemies
        self.n_allies = args.n_allies

    # Add new func, 5vs5
    def _get_obs_component_dim(self):
        own_feats_dim, ally_feats_dim, enemy_feats_dim  = self.args.obs_component  # [4, (5, 2), (4, 2)]
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        return (own_feats_dim, ally_feats_dim_flatten, enemy_feats_dim_flatten), (
            enemy_feats_dim, ally_feats_dim)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        obs_component_dim, _ = self._get_obs_component_dim()
        raw_obs_t = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        own_feats_t, ally_feats_t, enemy_feats_t = th.split(raw_obs_t, obs_component_dim, dim=-1)

        own_feats_t = own_feats_t.reshape(bs * self.n_agents, -1) # [bs * n_agents, fea_dim]
        ally_feats_t = ally_feats_t.reshape(bs * self.n_agents * self.n_allies,
                                            -1)  # [bs * n_agents * n_allies, a_fea_dim]
        enemy_feats_t = enemy_feats_t.reshape(bs * self.n_agents * self.n_enemies,
                                              -1)  # [bs * n_agents * n_enemies, fea_dim]

        own_inputs_t = [own_feats_t]
        if self.args.obs_agent_id:
            # agent-id indices, [bs, n_agents, n_agents]
            own_inputs_t.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        if self.args.obs_last_action:
            if t == 0:
                own_inputs_t.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                own_inputs_t.append(batch["actions_onehot"][:, t - 1])

        own_inputs_t = th.cat([x.reshape(bs * self.n_agents, -1) for x in own_inputs_t], dim=-1)

        return bs, own_inputs_t, ally_feats_t, enemy_feats_t

    def _get_input_shape(self, scheme):
        own_feats_dim, ally_feats_dim, enemy_feats_dim = self.args.obs_component
        own_input_shape = own_feats_dim
        ally_input_shape = copy.deepcopy(ally_feats_dim)
        enemy_input_shape = copy.deepcopy(enemy_feats_dim)

        if self.args.obs_agent_id:
            own_input_shape += self.n_agents
        if self.args.obs_last_action:
            own_input_shape += scheme["actions_onehot"]["vshape"][0]

        return own_input_shape, ally_input_shape, enemy_input_shape
