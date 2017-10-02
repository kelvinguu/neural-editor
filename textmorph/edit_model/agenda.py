import torch
from torch.nn import Module, Linear


class AgendaMaker(Module):
    def __init__(self, source_dim, edit_dim, agenda_dim):
        super(AgendaMaker, self).__init__()
        self.linear = Linear(source_dim + edit_dim, agenda_dim)

    def forward(self, source_embed, edit_embed):
        """Create agenda vector from source text embedding and edit embedding.

        Args:
            source_embed (Variable): of shape (batch_size, source_dim)
            edit_embed (Variable): of shape (batch_size, edit_dim)

        Returns:
            agenda (Variable): of shape (batch_size, agenda_dim)
        """
        inp = torch.cat([source_embed, edit_embed], 1)  # (batch_size, hidden_dim + edit_dim)
        return self.linear(inp)