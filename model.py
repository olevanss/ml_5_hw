import torch.nn as nn

class CycleGAN(nn.Module):
    def __init__(self, generator_channels=64, discriminator_channels=64, use_dropout=False):
        super(CycleGAN, self).__init__()
        # Генераторы
        self.gen_a2b = Generator(generator_channels, use_dropout=use_dropout)
        self.gen_b2a = Generator(generator_channels, use_dropout=use_dropout)
        # Дискриминаторы
        self.dis_a = Discriminator(discriminator_channels)
        self.dis_b = Discriminator(discriminator_channels)

    def forward(self, real_A, real_B):
        fake_B = self.gen_a2b(real_A)
        fake_A = self.gen_b2a(real_B)
        rec_A = self.gen_b2a(fake_B)
        rec_B = self.gen_a2b(fake_A)

        #identity_A = self.gen_b2a(real_A)
        #identity_B = self.gen_a2b(real_B)

        dis_a_real = self.dis_a(real_A)
        dis_a_fake = self.dis_a(fake_A)
        dis_b_real = self.dis_b(real_B)
        dis_b_fake = self.dis_b(fake_B)

        return {
            "fake_B": fake_B,
            "fake_A": fake_A,
            "rec_A": rec_A,
            "rec_B": rec_B,
            #"identity_A": identity_A,
            #f"identity_B": identity_B,
            "dis_a_real": dis_a_real,
            "dis_a_fake": dis_a_fake,
            "dis_b_real": dis_b_real,
            "dis_b_fake": dis_b_fake,
        }

class ResnetBlock(nn.Module):
    def __init__(self, channels, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            norm_layer(channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers += [nn.Dropout(0.5)]

        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            norm_layer(channels),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, generator_channels=32, n_res_blocks=4, use_dropout=False):
        super(Generator, self).__init__()
        input_channels=3
        output_channels=3
        norm_layer = nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, generator_channels, kernel_size=7),
            norm_layer(generator_channels),
            nn.ReLU(inplace=True)
        ]

        curr_channels = generator_channels
        for _ in range(2):
            model += [
                nn.Conv2d(curr_channels, curr_channels * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(curr_channels * 2),
                nn.ReLU(inplace=True)
            ]
            curr_channels *= 2

        for _ in range(n_res_blocks):
            model += [ResnetBlock(curr_channels, norm_layer, use_dropout)]

        for _ in range(2):
            model += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(curr_channels, curr_channels // 2, kernel_size=3, stride=1, padding=1),
                norm_layer(curr_channels // 2),
                nn.ReLU(inplace=True)
            ]
            curr_channels //= 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_channels, output_channels, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, discriminator_channels=32):
        super(Discriminator, self).__init__()
        norm_layer = nn.InstanceNorm2d

        model = [
            nn.Conv2d(3, discriminator_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        curr_channels = discriminator_channels
        for n in [discriminator_channels * 2, discriminator_channels * 4, discriminator_channels * 8]:
            stride = 1 if n == discriminator_channels * 8 else 2
            model += [
                nn.Conv2d(curr_channels, n, kernel_size=4, stride=stride, padding=1),
                norm_layer(n),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            curr_channels = n

        model += [nn.Conv2d(curr_channels, 1, kernel_size=4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)