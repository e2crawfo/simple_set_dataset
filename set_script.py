import numpy as np
from skimage.transform import resize
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pprint
import time
import imageio
import itertools
import pickle

import dps
from dps import cfg
from dps.utils import (
    image_to_string, Param, Parameterized, get_param_hash, NumpySeed, map_structure)
from dps.datasets import load_backgrounds, background_names


class PickleWriter:
    def __init__(self, filename):
        self.filename = filename
        self.data = []

    def write(self, **kwargs):
        self.data.append(kwargs)

    def close(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.data, f)


class PickleReader:
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'rb') as f:
            self.data = pickle.load(f)

    def sample(self, batch_size):
        indices = np.random.permutation(len(self.data))[:batch_size]
        sampled_data = [self.data[i] for i in indices]
        sampled_data = map_structure(
            lambda *arrays: np.array(arrays),
            *sampled_data,
            is_leaf=lambda item: not isinstance(item, dict))
        return sampled_data


class Dataset(Parameterized):
    n_examples = Param()
    seed = Param()

    def __init__(self, shuffle=True, **kwargs):
        start = time.time()
        print("Trying to find dataset in cache...")

        directory = kwargs.get(
            "data_dir",
            os.path.join(cfg.data_dir, "cached_datasets", self.__class__.__name__))
        os.makedirs(directory, exist_ok=True)

        params = self.param_values()
        param_hash = get_param_hash(params)
        print(self.__class__.__name__)
        print("Params:")
        pprint.pprint(params)
        print("Param hash: {}".format(param_hash))

        self.filename = os.path.join(directory, str(param_hash))
        cfg_filename = self.filename + ".cfg"

        no_cache = os.getenv("DPS_NO_CACHE")
        if no_cache:
            print("Skipping dataset cache as DPS_NO_CACHE is set (value is {}).".format(no_cache))

        # We require cfg_filename to exist as it marks that dataset creation completed successfully.
        if no_cache or not os.path.exists(self.filename) or not os.path.exists(cfg_filename):

            if kwargs.get("no_make", False):
                raise Exception("`no_make` is True, but dataset was not found in cache.")

            try:
                os.remove(self.filename)
            except FileNotFoundError:
                pass

            try:
                os.remove(cfg_filename)
            except FileNotFoundError:
                pass

            print("File for dataset not found, creating...")

            self._writer = PickleWriter(self.filename)
            try:
                with NumpySeed(self.seed):
                    self._make()

                self._writer.close()

                print("Done creating dataset.")
            except BaseException:
                self._writer.close()

                try:
                    os.remove(self.filename)
                except FileNotFoundError:
                    pass
                try:
                    os.remove(cfg_filename)
                except FileNotFoundError:
                    pass

                raise

            with open(cfg_filename, 'w') as f:
                f.write(pprint.pformat(params))
        else:
            print("Found.")

        print("Took {} seconds.\n".format(time.time() - start))

    def _make(self):
        raise Exception("AbstractMethod.")

    def _write_example(self, **kwargs):
        self._writer.write(**kwargs)


class ImageDataset(Dataset):
    image_shape = Param((100, 100))
    postprocessing = Param("")
    tile_shape = Param(None)
    n_samples_per_image = Param(1)

    @property
    def obs_shape(self):
        if self.postprocessing:
            return self.tile_shape + (self.depth,)
        else:
            return self.image_shape + (self.depth,)

    def _write_single_example(self, **kwargs):
        self._writer.write(**kwargs)

    def _write_example(self, **kwargs):
        image = kwargs['image']
        annotation = kwargs.get("annotations", [])
        label = kwargs.get("label", None)
        background = kwargs.get("background", None)

        if self.postprocessing == "tile":
            images, annotations, backgrounds = self._tile_postprocess(image, annotation, background=background)
        elif self.postprocessing == "random":
            images, annotations, backgrounds = self._random_postprocess(image, annotation, background=background)
        else:
            images, annotations, backgrounds = [image], [annotation], [background]

        for img, a, bg in itertools.zip_longest(images, annotations, backgrounds):
            self._write_single_example(image=img, annotations=a, label=label, background=bg)

    @staticmethod
    def tile_sample(image, tile_shape):
        height, width, n_channels = image.shape

        hangover = width % tile_shape[1]
        if hangover != 0:
            pad_amount = tile_shape[1] - hangover
            pad_shape = (height, pad_amount)
            padding = np.zeros(pad_shape)
            image = np.concat([image, padding], axis=2)

        hangover = height % tile_shape[0]
        if hangover != 0:
            pad_amount = tile_shape[0] - hangover
            pad_shape = list(image.shape)
            pad_shape[1] = pad_amount
            padding = np.zeros(pad_shape)
            image = np.concat([image, padding], axis=1)

        pad_height = tile_shape[0] - height % tile_shape[0]
        pad_width = tile_shape[1] - width % tile_shape[1]
        image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), 'constant')

        H = int(height / tile_shape[0])
        W = int(width / tile_shape[1])

        slices = np.split(image, W, axis=1)
        new_shape = (H, *tile_shape, n_channels)
        slices = [np.reshape(s, new_shape) for s in slices]
        new_images = np.concatenate(slices, axis=1)
        new_images = new_images.reshape(H * W, *tile_shape, n_channels)
        return new_images

    def _tile_postprocess(self, image, annotations, background=None):
        new_images = self.tile_sample(image, self.tile_shape)
        new_annotations = []

        H = int(image.shape[0] / self.tile_shape[0])
        W = int(image.shape[1] / self.tile_shape[1])

        for h in range(H):
            for w in range(W):
                offset = (h * self.tile_shape[0], w * self.tile_shape[1])
                _new_annotations = []
                for l, top, bottom, left, right in annotations:
                    # Transform to tile co-ordinates
                    top = top - offset[0]
                    bottom = bottom - offset[0]
                    left = left - offset[1]
                    right = right - offset[1]

                    # Restrict to chosen crop
                    top = np.clip(top, 0, self.tile_shape[0])
                    bottom = np.clip(bottom, 0, self.tile_shape[0])
                    left = np.clip(left, 0, self.tile_shape[1])
                    right = np.clip(right, 0, self.tile_shape[1])

                    invalid = (bottom - top < 1e-6) or (right - left < 1e-6)

                    if not invalid:
                        _new_annotations.append((l, top, bottom, left, right))

                new_annotations.append(_new_annotations)

        new_backgrounds = []
        if background is not None:
            new_backgrounds = self.tile_sample(background, self.tile_shape)

        return new_images, new_annotations, new_backgrounds

    def _random_postprocess(self, image, annotations, background=None):
        height, width, _ = image.shape
        new_images = []
        new_annotations = []
        new_backgrounds = []

        for j in range(self.n_samples_per_image):
            _top = np.random.randint(0, height-self.tile_shape[0]+1)
            _left = np.random.randint(0, width-self.tile_shape[1]+1)

            crop = image[_top:_top+self.tile_shape[0], _left:_left+self.tile_shape[1], ...]
            new_images.append(crop)

            if background is not None:
                bg_crop = background[_top:_top+self.tile_shape[0], _left:_left+self.tile_shape[1], ...]
                new_backgrounds.append(bg_crop)

            offset = (_top, _left)
            _new_annotations = []
            for l, top, bottom, left, right in annotations:
                top = top - offset[0]
                bottom = bottom - offset[0]
                left = left - offset[1]
                right = right - offset[1]

                top = np.clip(top, 0, self.tile_shape[0])
                bottom = np.clip(bottom, 0, self.tile_shape[0])
                left = np.clip(left, 0, self.tile_shape[1])
                right = np.clip(right, 0, self.tile_shape[1])

                invalid = (bottom - top < 1e-6) or (right - left < 1e-6)

                if not invalid:
                    _new_annotations.append((l, top, bottom, left, right))

            new_annotations.append(_new_annotations)

        return new_images, new_annotations, new_backgrounds

    def visualize(self, n=4):
        batch_size = n
        sample = self.sample(n)
        images = sample["image"]
        annotations = sample["annotations"]
        label = sample["label"]

        fig, axes = plt.subplots(1, batch_size)
        for i in range(batch_size):
            ax = axes[i]
            ax.set_axis_off()

            ax.imshow(np.squeeze(images[i]))
            ax.set_title("label={}".format(label[i]))

        plt.subplots_adjust(top=0.95, bottom=0, left=0, right=1, wspace=0.1, hspace=0.1)
        plt.show()

    def sample(self, n=4):
        reader = PickleReader(self.filename)
        _sample = reader.sample(n)
        return _sample


class Rectangle(object):
    def __init__(self, y, x, h, w):
        self.top = y
        self.bottom = y+h
        self.left = x
        self.right = x+w

        self.h = h
        self.w = w

    def intersects(self, r2):
        return self.overlap_area(r2) > 0

    def overlap_area(self, r2):
        overlap_bottom = np.minimum(self.bottom, r2.bottom)
        overlap_top = np.maximum(self.top, r2.top)

        overlap_right = np.minimum(self.right, r2.right)
        overlap_left = np.maximum(self.left, r2.left)

        area = np.maximum(overlap_bottom - overlap_top, 0) * np.maximum(overlap_right - overlap_left, 0)
        return area

    def centre(self):
        return (
            self.top + (self.bottom - self.top) / 2.,
            self.left + (self.right - self.left) / 2.
        )

    def __str__(self):
        return "Rectangle({}:{}, {}:{})".format(self.top, self.bottom, self.left, self.right)


class PatchesDataset(ImageDataset):
    max_overlap = Param(10)
    draw_shape = Param(None)
    draw_offset = Param((0, 0))
    patch_size_std = Param(None)
    distractor_shape = Param((3, 3))
    n_distractors_per_image = Param(0)
    backgrounds = Param(
        "", help="Can be either be 'all', in which a random background will be selected for "
                 "each constructed image, or a list of strings, giving the names of backgrounds "
                 "to use.")
    backgrounds_sample_every = Param(
        False, help="If True, sample a new sub-region of background for each image. Otherwise, "
                    "sample a small set of regions initially, and use those for all images.")
    backgrounds_resize = Param(False)
    background_colours = Param("")
    max_attempts = Param(10000)
    colours = Param('red green blue')

    one_hot = Param(True)
    plot_every = Param(None)

    @property
    def n_classes(self):
        raise Exception("AbstractMethod")

    @property
    def depth(self):
        return 3 if self.colours else 1

    def _make(self):
        if self.n_examples == 0:
            return np.zeros((0,) + self.image_shape).astype('uint8'), np.zeros((0, 1)).astype('i')

        # --- prepare colours ---

        colours = self.colours
        if colours is None:
            colours = []
        if isinstance(colours, str):
            colours = colours.split()

        colour_map = mpl.colors.get_named_colors_mapping()

        self._colours = []
        for c in colours:
            c = mpl.colors.to_rgb(colour_map[c])
            c = np.array(c)[None, None, :]
            c = np.uint8(255. * c)
            self._colours.append(c)

        # --- prepare shapes ---

        self.draw_shape = self.draw_shape or self.image_shape
        self.draw_offset = self.draw_offset or (0, 0)

        draw_shape = self.draw_shape
        if self.depth is not None:
            draw_shape = draw_shape + (self.depth,)

        # --- prepare backgrounds ---

        if self.backgrounds == "all":
            backgrounds = background_names()
        elif isinstance(self.backgrounds, str):
            backgrounds = self.backgrounds.split()
        else:
            backgrounds = self.backgrounds

        if backgrounds:
            if self.backgrounds_resize:
                backgrounds = load_backgrounds(backgrounds, draw_shape)
            else:
                backgrounds = load_backgrounds(backgrounds)

                if not self.backgrounds_sample_every:
                    _backgrounds = []
                    for b in backgrounds:
                        top = np.random.randint(b.shape[0] - draw_shape[0] + 1)
                        left = np.random.randint(b.shape[1] - draw_shape[1] + 1)
                        _backgrounds.append(
                            b[top:top+draw_shape[0], left:left+draw_shape[1], ...] + 0
                        )
                    backgrounds = _backgrounds

        background_colours = self.background_colours
        if isinstance(self.background_colours, str):
            background_colours = background_colours.split()
        _background_colours = []
        for bc in background_colours:
            color = mpl.colors.to_rgb(bc)
            color = np.array(color)[None, None, :]
            color = np.uint8(255. * color)
            _background_colours.append(color)
        background_colours = _background_colours

        # --- start dataset creation ---

        for j in range(int(self.n_examples)):
            if j % 1000 == 0:
                print("Working on datapoint {}...".format(j))

            # --- populate background ---

            if backgrounds:
                b_idx = np.random.randint(len(backgrounds))
                background = backgrounds[b_idx]
                if self.backgrounds_sample_every:
                    top = np.random.randint(background.shape[0] - draw_shape[0] + 1)
                    left = np.random.randint(background.shape[1] - draw_shape[1] + 1)
                    image = background[top:top+draw_shape[0], left:left+draw_shape[1], ...] + 0
                else:
                    image = background + 0
            elif background_colours:
                color = background_colours[np.random.randint(len(background_colours))]
                image = color * np.ones(draw_shape, 'uint8')
            else:
                image = np.zeros(draw_shape, 'uint8')

            # --- sample and populate patches ---

            locs, patches, patch_labels, image_label = self._sample_image()

            draw_offset = self.draw_offset

            for patch, loc in zip(patches, locs):
                if patch.shape[:2] != (loc.h, loc.w):
                    patch = resize(patch, (loc.h, loc.w), mode='edge', preserve_range=True)

                intensity = patch[:, :, :-1]
                alpha = patch[:, :, -1:].astype('f') / 255.

                current = image[loc.top:loc.bottom, loc.left:loc.right, ...]
                image[loc.top:loc.bottom, loc.left:loc.right, ...] = np.uint8(alpha * intensity + (1 - alpha) * current)

            # --- add distractors ---

            if self.n_distractors_per_image > 0:
                distractor_patches = self._sample_distractors()
                distractor_shapes = [img.shape for img in distractor_patches]
                distractor_locs = self._sample_patch_locations(distractor_shapes)

                for patch, loc in zip(distractor_patches, distractor_locs):
                    if patch.shape[:2] != (loc.h, loc.w):
                        patch = resize(patch, (loc.h, loc.w), mode='edge', preserve_range=True)

                    intensity = patch[:, :, :-1]
                    alpha = patch[:, :, -1:].astype('f') / 255.

                    current = image[loc.top:loc.bottom, loc.left:loc.right, ...]
                    image[loc.top:loc.bottom, loc.left:loc.right, ...] = np.uint8(alpha * intensity + (1 - alpha) * current)

            # --- possibly crop entire image ---

            if self.draw_shape != self.image_shape or draw_offset != (0, 0):
                image_shape = self.image_shape
                if self.depth is not None:
                    image_shape = image_shape + (self.depth,)

                draw_top = np.maximum(-draw_offset[0], 0)
                draw_left = np.maximum(-draw_offset[1], 0)

                draw_bottom = np.minimum(-draw_offset[0] + self.image_shape[0], self.draw_shape[0])
                draw_right = np.minimum(-draw_offset[1] + self.image_shape[1], self.draw_shape[1])

                image_top = np.maximum(draw_offset[0], 0)
                image_left = np.maximum(draw_offset[1], 0)

                image_bottom = np.minimum(draw_offset[0] + self.draw_shape[0], self.image_shape[0])
                image_right = np.minimum(draw_offset[1] + self.draw_shape[1], self.image_shape[1])

                _image = np.zeros(image_shape, 'uint8')
                _image[image_top:image_bottom, image_left:image_right, ...] = \
                    image[draw_top:draw_bottom, draw_left:draw_right, ...]

                image = _image

            annotations = self._get_annotations(draw_offset, patches, locs, patch_labels)

            self._write_example(image=image, annotations=annotations, label=image_label)

            if self.plot_every is not None and j % self.plot_every == 0:
                print(image_label)
                print(image_to_string(image))
                print("\n")
                plt.imshow(image)
                ax = plt.gca()

                for cls, top, bottom, left, right in annotations:
                    width = right - left
                    height = bottom - top

                    rect = mpl.patches.Rectangle(
                        (left, top), width, height, linewidth=1,
                        edgecolor='white', facecolor='none')

                    ax.add_patch(rect)

                plt.show()

    def _get_annotations(self, draw_offset, patches, locs, labels):
        new_labels = []
        for patch, loc, label in zip(patches, locs, labels):
            nz_y, nz_x = np.nonzero(patch[:, :, -1])

            # In draw co-ordinates
            top = (nz_y.min() / patch.shape[0]) * loc.h + loc.top
            bottom = (nz_y.max() / patch.shape[0]) * loc.h + loc.top
            left = (nz_x.min() / patch.shape[1]) * loc.w + loc.left
            right = (nz_x.max() / patch.shape[1]) * loc.w + loc.left

            # Transform to image co-ordinates
            top = top + draw_offset[0]
            bottom = bottom + draw_offset[0]
            left = left + draw_offset[1]
            right = right + draw_offset[1]

            top = np.clip(top, 0, self.image_shape[0])
            bottom = np.clip(bottom, 0, self.image_shape[0])
            left = np.clip(left, 0, self.image_shape[1])
            right = np.clip(right, 0, self.image_shape[1])

            invalid = (bottom - top < 1e-6) or (right - left < 1e-6)

            if not invalid:
                new_labels.append((label, top, bottom, left, right))

        return new_labels

    def _sample_image(self):
        patches, patch_labels, image_label = self._sample_patches()
        patch_shapes = np.array([img.shape for img in patches])

        locs = self._sample_patch_locations(
            patch_shapes,
            max_overlap=self.max_overlap,
            size_std=self.patch_size_std)
        return locs, patches, patch_labels, image_label

    def _sample_patches(self):
        raise Exception("AbstractMethod")

    def _sample_patch_locations(self, patch_shapes, max_overlap=None, size_std=None):
        """ Sample random locations within draw_shape. """
        if len(patch_shapes) == 0:
            return []

        patch_shapes = np.array(patch_shapes)
        n_rects = patch_shapes.shape[0]

        n_tries_outer = 0
        while True:
            rects = []
            for i in range(n_rects):
                n_tries_inner = 0
                while True:
                    if size_std is None:
                        shape_multipliers = 1.
                    else:
                        shape_multipliers = np.maximum(np.random.randn(2) * size_std + 1.0, 0.5)

                    m, n = np.ceil(shape_multipliers * patch_shapes[i, :2]).astype('i')

                    rect = Rectangle(
                        np.random.randint(0, self.draw_shape[0]-m+1),
                        np.random.randint(0, self.draw_shape[1]-n+1), m, n)

                    if max_overlap is None:
                        rects.append(rect)
                        break
                    else:
                        overlap_area = 0
                        violation = False
                        for r in rects:
                            overlap_area += rect.overlap_area(r)
                            if overlap_area > max_overlap:
                                violation = True
                                break

                        if not violation:
                            rects.append(rect)
                            break

                    n_tries_inner += 1

                    if n_tries_inner > self.max_attempts/10:
                        break

                if len(rects) < i + 1:  # No rectangle found
                    break

            if len(rects) == n_rects:
                break

            n_tries_outer += 1

            if n_tries_outer > self.max_attempts:
                raise Exception(
                    "Could not fit rectangles. "
                    "(n_rects: {}, draw_shape: {}, max_overlap: {})".format(
                        n_rects, self.draw_shape, max_overlap))

        return rects

    def _sample_distractors(self):
        distractor_images = []

        patches = []
        while not patches:
            patches, y, _ = self._sample_patches()

        for i in range(self.n_distractors_per_image):
            idx = np.random.randint(len(patches))
            patch = patches[idx]
            m, n, *_ = patch.shape
            source_y = np.random.randint(0, m-self.distractor_shape[0]+1)
            source_x = np.random.randint(0, n-self.distractor_shape[1]+1)

            img = patch[
                source_y:source_y+self.distractor_shape[0],
                source_x:source_x+self.distractor_shape[1]]

            distractor_images.append(img)

        return distractor_images

    def _colourize(self, img, colour=None):
        """ Apply a colour to a gray-scale image. """

        if isinstance(colour, str):
            colour = mpl.colors.to_rgb(colour)
            colour = np.array(colour)[None, None, :]
            colour = np.uint8(255. * colour)
        else:
            if colour is None:
                colour = np.random.randint(len(self._colours))
            colour = self._colours[int(colour)]

        rgb = np.tile(colour, img.shape + (1,))
        alpha = img[:, :, None]

        return np.concatenate([rgb, alpha], axis=2).astype(np.uint8)


class SetThreeAttr(PatchesDataset):
    colours = Param()
    shapes = Param()
    digits = Param()
    digit_colour = Param()
    n_cards = Param()
    set_size = Param()
    patch_shape = Param()

    n_classes = 2

    @staticmethod
    def _generate_cards_and_label(cards, n_cards, set_size):
        shuffled_cards = np.random.permutation(cards)
        drawn_cards = [tuple(c) for c in shuffled_cards[:n_cards]]

        for _set in itertools.combinations(drawn_cards, set_size):
            is_set = True
            for attr_idx in range(len(_set[0])):
                attr_values = set(card[attr_idx] for card in _set)

                if len(attr_values) == 1 or len(attr_values) == set_size:
                    continue
                else:
                    is_set = False
                    break

            if is_set:
                return drawn_cards, _set

        return drawn_cards, None

    def _get_patch_for_card(self, card):
        patch = self.patches.get(card, None)
        if patch is None:
            colour, shape, digit = card

            f = os.path.join(os.path.dirname(dps.__file__), "shapes", "{}.png".format(shape))
            image = imageio.imread(f)
            image = resize(image[..., 3], self.patch_shape, mode='edge', preserve_range=True)
            image = self._colourize(image, colour)

            shape_colour = image[:, :, :3]
            shape_alpha = image[:, :, 3:4] / 255.

            f = os.path.join(os.path.dirname(dps.__file__), "digits", "{}.png".format(digit))
            digit_image = imageio.imread(f)
            digit_image = resize(digit_image[..., 3], self.patch_shape, mode='edge', preserve_range=True)
            digit_image = self._colourize(digit_image, self.digit_colour)

            digit_colour = digit_image[:, :, :3]
            digit_alpha = digit_image[:, :, 3:4] / 255.

            image_rgb = digit_alpha * digit_colour + (1-digit_alpha) * shape_colour
            image_alpha = (np.clip(digit_alpha + shape_alpha, 0, 1) * 255).astype(np.uint8)

            patch = self.patches[card] = np.concatenate([image_rgb, image_alpha], axis=2)

        return patch

    def _make(self):
        if isinstance(self.colours, str):
            self.colours = self.colours.split()
        if isinstance(self.shapes, str):
            self.shapes = self.shapes.split()
        if isinstance(self.digits, str):
            self.digits = self.digits.split()

        self.cards = list(itertools.product(self.colours, self.shapes, self.digits))
        self.patches = {}
        self.n_pos = 0
        self.n_neg = 0
        super(SetThreeAttr, self)._make()

    def _sample_patches(self):
        label = np.random.randint(2)
        blabel = bool(label)

        is_set = not blabel
        while is_set != blabel:
            cards, _set = self._generate_cards_and_label(self.cards, self.n_cards, self.set_size)
            is_set = bool(_set)
            if is_set:
                self.n_pos += 1
            else:
                self.n_neg += 1

        patches = [self._get_patch_for_card(card) for card in cards]
        return patches, [0] * len(patches), label


if __name__ == "__main__":
    cfg.data_dir = "./dps_data/data"

    dset = SetThreeAttr(
        n_examples=16,
        background_colours="cyan magenta yellow",
        colours="red green blue",
        shapes="circle square diamond",
        digits="simple1 simple2 simple3",
        digit_colour="black",
        n_cards=7,
        set_size=3,
        seed=1001,
        image_shape=(48, 48),
        patch_shape=(14, 14),
        max_overlap=14*14/3)

    dset.visualize(n=16)
