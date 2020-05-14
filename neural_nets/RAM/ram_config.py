class RAM_Config(object):
    def __init__(self, batch_size, max_epochs, num_classes, nGlimpses=6, translated=False):

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_iters = 10000

        self.nGlimpses = nGlimpses
        # paramters about the training examples
        self.n_classes = num_classes

        self.save_dir = "chckPts/"
        self.save_prefix = "save"
        self.summaryFolderName = "summary/"

        self.start_step = 0  # config.startstep
        self.model_name = "tchng_dgts_centered_RAM"
        self.ckpt_path = self.save_dir + self.save_prefix + self.model_name + ".ckpt"
        self.meta_path = self.save_dir + self.save_prefix + self.model_name + ".ckpt.meta"

        # conditions
        self.eyeCentered = 0

        # about translation
        self.ORIG_IMG_SIZE = 28
        self.translated_img_size = 128  # side length of the picture

        self.fixed_learning_rate = 0.001

        # config("max_iters")
        self.SMALL_NUM = 1e-10
        # config("small_num")

        self.translateMnist = None
        self.img_size = None
        self.depth = 0
        self.sensorBandwidth = 0
        self.minRadius = None

        self.initLr = None
        self.lr_min = None
        self.lrDecayRate = None
        self.lrDecayFreq = None
        self.momentumValue = None

        if translated:
            self.init_translated()
        else:
            self.init_original()

        self.channels = 1  # mnist are grayscale images
        self.totalSensorBandwidth = self.depth * self.channels * (self.sensorBandwidth ** 2)
        self.loc_sd = 0.15  # std when setting the location

        # network units
        self.hg_size = 128  #
        self.hl_size = 128  #
        self.g_size = 256  #
        self.cell_size = 256 #
        self.cell_out_size = self.cell_size  #

    def init_translated(self):
        self.translateMnist = 1
        self.img_size = self.translated_img_size
        self.depth = 3  # number of zooms
        self.sensorBandwidth = 20
        self.minRadius = 12  # zooms -> minRadius * 2**<depth_level>

        self.initLr = 1e-3
        self.lr_min = 1e-4
        self.lrDecayRate = 0.999
        self.lrDecayFreq = 200
        self.momentumValue = 0.9

    def init_original(self):
        self.translateMnist = 0
        self.img_size = self.ORIG_IMG_SIZE
        self.depth = 2 # number of zooms
        self.sensorBandwidth = 16
        self.minRadius = 4  # zooms -> minRadius * 2**<depth_level>

        self.initLr = 0.0011
        self.lrDecayRate = .99
        self.lrDecayFreq = 200
        self.momentumValue = .95
