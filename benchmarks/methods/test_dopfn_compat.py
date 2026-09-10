"""DoPFN check_array compatibility across old and new scikit-learn APIs."""
import importlib.util
import inspect
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

# Load compiled dependencies before patch.dict snapshots sys.modules. They
# cannot be unloaded and imported again between fake sklearn environments.
import numpy  # noqa: F401
import torch  # noqa: F401


class CheckArrayCompatibilityTest(unittest.TestCase):
    def load_shim(self, check_array):
        sklearn = types.ModuleType('sklearn')
        utils = types.ModuleType('sklearn.utils')
        validation = types.ModuleType('sklearn.utils.validation')
        sklearn.utils, utils.validation = utils, validation
        utils.check_array = validation.check_array = check_array
        modules = {'sklearn': sklearn, 'sklearn.utils': utils,
                   'sklearn.utils.validation': validation}
        context = patch.dict(sys.modules, modules)
        context.start()
        self.addCleanup(context.stop)
        spec = importlib.util.spec_from_file_location(
            '_dopfn_compat_test', Path(__file__).with_name('dopfn.py'))
        shim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shim)
        return shim, utils, validation

    def test_old_keyword_on_modern_sklearn(self):
        def modern(array, *, ensure_all_finite=True, dtype=None):
            return array, ensure_all_finite, dtype

        _, utils, validation = self.load_shim(modern)
        # The exact call shape in DoPFN.check_training_data on Killarney.
        self.assertEqual(utils.check_array([1], force_all_finite=False, dtype='float32'),
                         ([1], False, 'float32'))
        self.assertEqual(validation.check_array([1], ensure_all_finite='allow-nan'),
                         ([1], 'allow-nan', None))
        self.assertEqual(utils.check_array([1]), ([1], True, None))
        self.assertEqual(inspect.signature(utils.check_array), inspect.signature(modern))

    def test_new_keyword_on_legacy_sklearn(self):
        def legacy(array, *, force_all_finite=True):
            return force_all_finite

        _, utils, validation = self.load_shim(legacy)
        self.assertIs(utils.check_array([1], ensure_all_finite=False), False)
        self.assertEqual(validation.check_array([1], ensure_all_finite='allow-nan'), 'allow-nan')
        self.assertIs(utils.check_array([1], force_all_finite=False), False)
        self.assertIs(utils.check_array([1]), True)

    def test_transition_api_is_left_intact(self):
        def transition(array, *, force_all_finite='deprecated', ensure_all_finite=None):
            return force_all_finite, ensure_all_finite

        shim, utils, validation = self.load_shim(transition)
        shim._repatch_dopfn_check_array()
        self.assertIs(utils.check_array, transition)
        self.assertIs(validation.check_array, transition)

    def test_repeated_installation_updates_late_import_without_nesting(self):
        for keyword in ('ensure_all_finite', 'force_all_finite'):
            with self.subTest(keyword=keyword):
                # Explicit signatures reproduce both actual sklearn APIs.
                namespace = {}
                exec(f'def check_array(array, *, {keyword}=True): return {keyword}', namespace)
                original = namespace['check_array']
                shim, utils, validation = self.load_shim(original)
                patched = utils.check_array
                base = types.ModuleType('scripts.transformer_prediction_interface.base')
                base.check_array = original
                with patch.dict(sys.modules, {base.__name__: base}):
                    for _ in range(3):
                        shim._repatch_dopfn_check_array()
                        self.assertIs(utils.check_array, patched)
                        self.assertIs(validation.check_array, patched)
                        self.assertIs(base.check_array, patched)
                        other = ('force_all_finite' if keyword == 'ensure_all_finite'
                                 else 'ensure_all_finite')
                        self.assertIs(base.check_array([1], **{other: False}), False)

    def test_conflicting_keywords_are_not_silently_overwritten(self):
        def modern(array, *, ensure_all_finite=True):
            return ensure_all_finite

        _, utils, _ = self.load_shim(modern)
        with self.assertRaisesRegex(TypeError, 'both'):
            utils.check_array([1], force_all_finite=False, ensure_all_finite=True)


if __name__ == '__main__':
    unittest.main()
