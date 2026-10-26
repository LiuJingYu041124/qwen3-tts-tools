import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.io import wavfile
import tts_worker as w

class CompletionGateTests(unittest.TestCase):
    def result(self):
        return iter([SimpleNamespace(audio=np.ones(24000,dtype=np.float32)*.1,sample_rate=24000,token_count=12)])

    def test_collect_preserves_all_chunks(self):
        a,s,t=w.collect_audio(iter([SimpleNamespace(audio=np.array([.1,.2]),sample_rate=24000,token_count=2),SimpleNamespace(audio=np.array([.3]),sample_rate=24000,token_count=1)]))
        np.testing.assert_allclose(a,[.1,.2,.3]); self.assertEqual(t,3)

    def test_terminal_failure_is_not_returned_as_audio(self):
        with patch.object(w,'generate_cue_results',side_effect=lambda *a:self.result()), patch.object(w,'terminal_particle_present',return_value=(False,'这是一直延伸的')):
            with self.assertRaisesRegex(RuntimeError,'没有产出可用音频'):
                w.generate_audio(None,'design',{},dict(id=1,role='r',text='这是一直延伸的吗？'),'Chinese',1)

    def test_clone_terminal_particle_failure_is_retained_for_review(self):
        cue = {'id': 30, 'role': 'role4', 'text': '我只是想玩一下啊！', 'emotion': '略带思考、语气平稳'}
        with patch.object(w, 'generate_cue_results', side_effect=lambda *a: self.result()), \
                patch.object(w, 'terminal_particle_present', return_value=(False, '我只是想玩一下')), \
                patch.object(w, 'transcribe_full_cue', return_value='我只是想玩一下'), \
                patch.object(w, 'clone_pronunciation_variants', return_value=[]):
            result = w.generate_audio(
                None, 'clone', {'id': 'role4'}, cue, 'Chinese', 930079, review_sink={}
            )
        self.assertTrue(result[5]['best_effort'])
        self.assertEqual(result[5]['gate_transcript'], '我只是想玩一下')
        self.assertFalse(result[5]['gate_quality']['passed'])

    def test_traditional_terminal_resumes_and_reuses_completed_track_cues(self):
        cue = {'id': 45, 'role': 'role1', 'text': '方块上刻的太阳图案……这一关会和太阳有关吗？'}
        transcript = '方塊上刻的太陽圖案這一關會和太陽有關嗎?'
        previous = {'id': 44, 'role': 'role1', 'text': cue['text']}
        voices = {'role1': {'id': 'role1', 'mode': 'clone'}}
        models = {'design': 'design', 'clone': 'clone'}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'raw').mkdir()
            w.write_json_atomic(root / 'manifest.json', [previous, cue])
            w.write_json_atomic(root / 'voices.json', voices)
            cached_audio = root / 'raw' / '044-role1.wav'
            wavfile.write(cached_audio, 24000, np.full(24000, .1, dtype=np.float32))
            w.write_json_atomic(cached_audio.with_suffix('.json'), {
                'fingerprint': w.raw_cache_key(previous, voices['role1'], 'Chinese', models, 'natural'),
                'asr_quality': w.cue_asr_quality(previous['text'], transcript),
            })
            protected = (cached_audio.read_bytes(), cached_audio.with_suffix('.json').read_bytes())
            failure = root / 'raw' / '045-role1.failed.json'
            w.write_json_atomic(failure, {'error': '句尾完整性校验失败'})
            args = SimpleNamespace(
                root=str(root), design_model='design', clone_model='clone', language='zh',
                timeline_mode='natural', reuse=True, gap=0.25, duration=0,
            )

            def copy_audio(source, target, *args):
                Path(target).write_bytes(Path(source).read_bytes())

            with patch.object(w, 'load_model', return_value=object()) as load, \
                    patch.object(w, 'generate_cue_results', side_effect=lambda *a: self.result()) as generate, \
                    patch.object(w, 'transcribe_audio_array', return_value=transcript), \
                    patch.object(w, 'clean_audio', side_effect=copy_audio), \
                    patch.object(w, 'fit_natural_audio', side_effect=copy_audio):
                w.build_project(args)
                self.assertEqual(generate.call_count, 1)
                self.assertEqual(load.call_count, 1)
                self.assertFalse(failure.exists())
                self.assertEqual(protected, (cached_audio.read_bytes(), cached_audio.with_suffix('.json').read_bytes()))
                summary = w.read_json(root / 'summary.json')
                self.assertEqual(summary['cues'], 2)
                self.assertEqual(summary['verified_cues'], 2)
                self.assertEqual(summary['needs_review'], 0)
                self.assertEqual(summary['raw_cache_hits'], 1)
                self.assertEqual(summary['generation_qa_reuses'], 1)
                self.assertEqual(summary['generation_work_cache']['candidate_executions'], 1)
                self.assertEqual([row['id'] for row in w.read_json(root / 'status.json')], [44, 45])
                track_before = (root / 'voice_timeline.wav').read_bytes()
                self.assertGreater(w.duration(root / 'voice_timeline.wav'), 2)
                with patch.object(w, 'transcribe_audio_array', side_effect=AssertionError('应复用ASR缓存')):
                    w.build_project(args)
                self.assertEqual(generate.call_count, 1)
                self.assertEqual(load.call_count, 1)
                reused_summary = w.read_json(root / 'summary.json')
                self.assertEqual(reused_summary['raw_cache_hits'], 2)
                self.assertEqual(reused_summary['generation_qa_reuses'], 0)
                self.assertEqual(sum(reused_summary['generation_work_cache'].values()), 0)
                self.assertEqual((root / 'voice_timeline.wav').read_bytes(), track_before)

    def test_traditional_terminal_preview_passes_without_best_effort(self):
        text = '方块上刻的太阳图案……这一关会和太阳有关吗？'
        transcript = '方塊上刻的太陽圖案這一關會和太陽有關嗎?'
        for mode in ('clone', 'design'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                request = root / 'preview.json'
                output = root / 'preview.wav'
                w.write_json_atomic(request, {
                    'voice': {'id': 'role1', 'mode': mode}, 'text': text, 'language': 'zh',
                })
                args = SimpleNamespace(preview_request=str(request), preview_output=str(output),
                                       design_model='design', clone_model='clone')
                with patch.object(w, 'load_model', return_value=object()), \
                        patch.object(w, 'generate_cue_results', side_effect=lambda *a: self.result()) as generate, \
                        patch.object(w, 'transcribe_audio_array', return_value=transcript), \
                        patch.object(w, 'clean_audio', side_effect=lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes())):
                    w.build_preview(args)
                self.assertEqual(generate.call_count, 1)
                self.assertTrue(output.is_file())
                self.assertFalse(output.with_suffix('.raw.wav').exists())
                self.assertFalse(output.with_suffix('.review.json').exists())

    def test_failed_whole_cue_is_not_published(self):
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'x.wav'; meta=Path(d)/'x.json'; cue=dict(id=1,role='r',text='我站在岩浆池旁边。')
            selected=(np.ones(24000,dtype=np.float32)*.1,24000,{},None,12)
            with patch.object(w,'load_model',return_value=object()),patch.object(w,'generate_audio',return_value=selected),patch.object(w,'transcribe_audio_array',return_value='我站在岩浆池'):
                w.load_and_generate('test','design',[(cue,{},raw,meta,'test',{})],'Chinese',{'design':'test'},'natural')
            self.assertFalse(raw.exists()); self.assertFalse(meta.exists())
            failure = raw.parent / '001-r.failed.json'
            self.assertTrue(failure.exists())
            self.assertIn('没有产出可发布音频', json.loads(failure.read_text())['error'])

    def test_incomplete_build_never_publishes_timeline_or_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cues = [
                {'id': 1, 'role': 'r', 'text': '第一句', 'enabled': True},
                {'id': 2, 'role': 'r', 'text': '第二句', 'enabled': True},
            ]
            (root / 'manifest.json').write_text(json.dumps(cues), encoding='utf-8')
            (root / 'voices.json').write_text(json.dumps({'r': {'id': 'r', 'mode': 'design'}}), encoding='utf-8')
            (root / 'voice_timeline.wav').write_bytes(b'stale-timeline')
            (root / 'summary.json').write_text(json.dumps({'cues': 2}), encoding='utf-8')
            args = SimpleNamespace(
                root=str(root), design_model='design', clone_model='clone', language='zh',
                timeline_mode='natural', reuse=True, gap=0.25, duration=0,
            )
            with patch.object(w, 'load_and_generate'):
                with self.assertRaisesRegex(RuntimeError, r'整轨构建未完成：0/2，这些台词没生成出音频 \[1, 2\]'):
                    w.build_project(args)
            self.assertFalse((root / 'voice_timeline.wav').exists())
            self.assertFalse((root / 'summary.json').exists())
            self.assertEqual(json.loads((root / 'status.json').read_text()), [])

    def test_comma_segment_retries_missing_tail_before_joining(self):
        with patch.object(w,'generate_cue_results',side_effect=lambda *a:self.result()) as gen, patch.object(w,'transcribe_audio_array',side_effect=['我反复比对图','我反复比对图片','当时站在旁边','我反复比对图片当时站在旁边']):
            w.generate_audio(None,'design',{},dict(id=12,role='r',text='我反复比对图片，当时站在旁边。'),'Chinese',1)
        self.assertEqual(gen.call_count,3)

    def test_comma_segment_failure_is_not_joined(self):
        with patch.object(w,'generate_cue_results',side_effect=lambda *a:self.result()), patch.object(w,'transcribe_audio_array',return_value='我反复比对图'):
            with self.assertRaisesRegex(RuntimeError,'没有产出可用音频'):
                w.generate_audio(None,'design',{},dict(id=12,role='r',text='我反复比对图片，当时站在旁边。'),'Chinese',1)

    def test_failed_and_unverified_cache_not_approved(self):
        self.assertFalse(w.raw_cache_approved({'asr_quality': {'passed': False}}))
        self.assertFalse(w.raw_cache_approved({}))
        self.assertTrue(w.raw_cache_approved({'asr_quality': {'passed': True}}))

    def test_clone_pronunciation_variants_cover_verified_inputs(self):
        self.assertEqual(
            dict(w.clone_pronunciation_variants('嘿嘿！放心吧，我明白！', 'Chinese')),
            {
                'clone-interjection-comma': '嘿嘿，放心吧，我明白！',
                'clone-interjection-period': '嘿嘿。放心吧，我明白！',
            },
        )
        self.assertEqual(
            dict(w.clone_pronunciation_variants('啊！阿芙，它飞下来了！', 'Chinese')),
            {
                'clone-interjection-period': '啊。阿芙，它飞下来了！',
                'clone-interjection-comma': '啊，阿芙，它飞下来了！',
            },
        )
        self.assertEqual(
            w.clone_pronunciation_performance_variants(
                'clone-interjection-comma', '嘿嘿，放心吧，我明白！',
            ),
            [('clear-leading-laugh', '轻快、含着笑，句首两个“嘿”清晰发音，不要吞掉')],
        )
        self.assertEqual(
            w.clone_pronunciation_performance_variants(
                'clone-interjection-period', '啊。阿芙，它飞下来了！',
            ),
            [('clear-leading-interjection', '激动、语气上扬，句首“啊”清晰发音，不要吞掉')],
        )
        self.assertEqual(
            w.clone_pronunciation_performance_variants(
                'clone-name-pair-截', '截，截，你快醒醒，千万不要出事啊！',
            ),
            [('clear-name-pair', '自然、清晰、平稳，名字两个音节分开完整发音，不要读成姐姐，不要添加语气词')],
        )
        self.assertEqual(
            dict(w.clone_pronunciation_variants(
                '阿远打算联系管理员反馈Bug，一秒，服务器强制关闭，视频就这样结束了。',
                'Chinese',
            ))['clone-name-boundary'],
            '阿远，打算联系管理员反馈Bug，一秒，服务器强制关闭，视频就这样结束了。',
        )
        self.assertEqual(
            dict(w.clone_pronunciation_variants(
                '阿远有事得离开，录制到这就结束了。',
                'Chinese',
            ))['clone-name-boundary'],
            '阿远，有事得离开，录制到这就结束了。',
        )
        self.assertEqual(
            dict(w.clone_pronunciation_variants(
                '这串字符看上去像乱码，但将他粘贴到视频网站上，直接指向了小宇自己的频道！',
                'Chinese',
            ))['clone-technical-boundary'],
            '这串，字符看上去像乱码，但将他粘贴到视频网站上，直接指向了小宇自己的频道！',
        )
        variants = dict(w.clone_pronunciation_variants(
            '杰杰，你快醒醒，千万不要出事啊！', 'Chinese',
        ))
        self.assertEqual(variants['clone-name-pair-截'], '截，截，你快醒醒，千万不要出事啊！')
        self.assertEqual(
            dict(w.clone_pronunciation_variants('杰妹，是你救了我吗？谢谢你。', 'Chinese'))['clone-name-companion-截'],
            '截妹，是你救了我吗？谢谢你。',
        )
        self.assertEqual(
            dict(w.clone_pronunciation_variants('小宇的屏幕立刻闪了一下。', 'Chinese'))['clone-name-tone-boundary'],
            '小，宇的屏幕立刻闪了一下。',
        )
        self.assertEqual(
            dict(w.clone_pronunciation_variants('身处其中的小宇，却毫无察觉？', 'Chinese'))['clone-placement-tone-boundary'],
            '身，处其中的小宇，却毫无察觉？',
        )
        self.assertEqual(
            dict(w.clone_pronunciation_variants('开头是Hyg什么opol....', 'Chinese'))['clone-latin-letter-boundary'],
            '开头是H，y，g，什么o，p，o，l。',
        )
        self.assertEqual(
            dict(w.clone_pronunciation_variants('啊~，糟了！小海有危险！', 'Chinese'))['clone-tilde-comma'],
            '啊，糟了！小海有危险！',
        )
        self.assertEqual(
            dict(w.clone_pronunciation_variants('这是苦力怕卡片……', 'Chinese'))['clone-terminal-period'],
            '这是苦力怕卡片。',
        )

    def test_clause_final_neutral_particle_pair_is_not_a_wrong_word(self):
        # Live clone runs read the identical audio back as 了 at one boundary and
        # 啦 at another, so the clause-final neutral pair must not fail the gate.
        for expected, transcript in (
            ('太棒啦！地图范围扩大到百分之百了！', '太棒啦!地图范围扩大到百分之百啦!'),
            ('绝对不能丢，只能用手拿着它了！', '绝对不能丢!只能用手拿着它啦!'),
            ('好了，火堆烧旺了！', '好啦!火堆燒忘了!'),
            ('小意思，我先走啦，拜拜！', '小意思我先走了拜拜'),
        ):
            self.assertTrue(w.cue_asr_quality(expected, transcript)['passed'], transcript)

    def test_clause_final_particle_rule_does_not_relax_other_syllables(self):
        for expected, transcript in (
            ('这是一直延伸的吗？', '这是一直延伸的猫'),
            ('了解了情况再说。', '啦解了情况再说。'),
            ('好了，别闹了！', '好啦，别弄了！'),
            ('啊！阿芙，它飞下来了！', '阿福他飞下来了'),
        ):
            self.assertFalse(w.cue_asr_quality(expected, transcript)['passed'], transcript)

    def test_clone_repeat_and_compound_variants_cover_verified_inputs(self):
        variants = dict(w.clone_pronunciation_variants(
            '大家加油！金金，你去河边钓几条鱼回来！', 'Chinese',
        ))
        self.assertEqual(
            variants['clone-repeat-boundary-金'],
            '大家加油！金，金，你去河边钓几条鱼回来！',
        )
        self.assertEqual(
            dict(w.clone_pronunciation_variants(
                '太好了！馅料有了，可是还缺少辣椒！', 'Chinese',
            ))['clone-hard-word-馅料'],
            '太好了！馅，料有了，可是还缺少辣椒！',
        )
        name_variants = dict(w.clone_pronunciation_variants(
            '杰杰还在养病，我要不要把这件事告诉他呢？', 'Chinese',
        ))
        self.assertEqual(
            name_variants['clone-name-pair-杰'],
            '杰，杰还在养病，我要不要把这件事告诉他呢？',
        )
        self.assertEqual(
            dict(w.clone_pronunciation_variants('杰妹？她要去哪里？', 'Chinese'))['clone-name-companion-杰'],
            '杰，妹？她要去哪里？',
        )
        # Ordinary reduplicated words must not be broken apart.
        self.assertNotIn(
            'clone-repeat-boundary-妈',
            dict(w.clone_pronunciation_variants('妈妈说明天见。', 'Chinese')),
        )
        self.assertEqual(
            w.clone_pronunciation_performance_variants('clone-repeat-boundary-金', '金，金'),
            [('clear-repeat-syllable',
              '自然、清晰、平稳，重复的两个音节都完整发音，中间有极短的停顿，不要合并成一个音或改成别的音')],
        )

    def test_clone_name_boundary_variant_covers_name_led_cues(self):
        self.assertEqual(
            dict(w.clone_pronunciation_variants(
                '阿远打了小宇一拳想吸引他注意，但小宇僵住了。',
                'Chinese',
            ))['clone-name-segmented'],
            '阿远，打了小宇一拳想吸引他注意，但小宇僵住了。',
        )
        self.assertEqual(
            dict(w.clone_pronunciation_variants(
                '阿远有事得离开，录制到这就结束了。',
                'Chinese',
            ))['clone-name-boundary'],
            '阿远，有事得离开，录制到这就结束了。',
        )

    def test_clone_embedded_name_boundary_variant_covers_long_cues(self):
        variants = dict(w.clone_pronunciation_variants(
            '服务器里很热闹，阿远主动和他打了招呼，可小宇的屏幕立刻就闪了一下。',
            'Chinese',
        ))
        self.assertEqual(
            variants['clone-embedded-name-boundary'],
            '服务器里很热闹，阿远，主动和他打了招呼，可小宇的屏幕立刻就闪了一下。',
        )

    def test_clone_context_retry_returns_verified_candidate(self):
        cue = {
            'id': 22,
            'role': 'role1',
            'text': '阿远有事得离开，录制到这就结束了。',
        }
        result = (np.ones(24000, dtype=np.float32) * .1, 24000, {'temperature': .72}, None, 12)
        seen = []

        def candidate(*args):
            seen.append(dict(args[3]))
            return result

        def transcript(*_):
            return cue['text'] if seen[-1].get('_context_review') else '阿元有事得离开录制到这就结束了'

        with patch.object(w, '_generate_audio_candidate', side_effect=candidate), \
                patch.object(w, 'transcribe_full_cue', side_effect=transcript):
            selected = w.generate_audio(
                None, 'clone', {'id': 'role1'}, cue, 'Chinese', 930862, 'natural',
            )
        self.assertEqual(selected[:5], result)
        self.assertTrue(any(item.get('_context_review') for item in seen))

    def test_clone_segmented_name_retry_preserves_phrase_boundaries(self):
        cue = {
            'id': 22,
            'role': 'role1',
            'text': '阿远打了小宇一拳想吸引他注意，但小宇僵住了。',
        }
        result = (np.ones(24000, dtype=np.float32) * .1, 24000, {'temperature': .72}, None, 12)
        seen = []

        def candidate(*args):
            seen.append(dict(args[3]))
            return result

        def transcript(*_):
            if seen[-1].get('_generation_strategy') == 'clone-name-segmented':
                return '阿远打了小羽一拳想吸引她注意但小羽僵住了'
            return '阿元打了小羽一拳想吸引她注意但小羽僵住了'

        with patch.object(w, '_generate_audio_candidate', side_effect=candidate), \
                patch.object(w, 'transcribe_full_cue', side_effect=transcript):
            selected = w.generate_audio(
                None, 'clone', {'id': 'role1'}, cue, 'Chinese', 930862, 'natural',
            )
        self.assertEqual(selected[:5], result)
        self.assertTrue(any(
            item.get('_generation_strategy') == 'clone-name-segmented'
            and not item.get('_context_review')
            for item in seen
        ))

    def test_clone_fallback_rejection_is_reported_after_all_candidates(self):
        cue = {'id': 95, 'role': 'role2', 'text': '杰杰！'}
        with patch.object(w, '_generate_audio_candidate', side_effect=RuntimeError('时长异常')):
            with self.assertRaisesRegex(RuntimeError, r'生成失败：.*最后候选被拒绝：时长异常'):
                w.generate_audio(None, 'clone', {'id': 'role2'}, cue, 'Chinese', 930862, 'natural')

    def test_duration_rejection_continues_to_clone_pronunciation_fallback(self):
        cue = {
            'id': 38,
            'role': 'role2',
            'text': '啊~，糟了！小海有危险！',
        }
        result = (np.ones(24000, dtype=np.float32) * .1, 24000, {'temperature': .72}, None, 12)
        seen = []

        def candidate(*args):
            seen.append(dict(args[3]))
            if args[3].get('_generation_strategy') is None:
                raise RuntimeError('克隆结果时长异常')
            return result

        def transcript(*_):
            return cue['text'] if seen[-1].get('_performance_strategy') == 'restrained-boundary' else '糟了小海有危险'

        with patch.object(w, '_generate_audio_candidate', side_effect=candidate), \
                patch.object(w, 'transcribe_full_cue', side_effect=transcript):
            selected = w.generate_audio(None, 'clone', {'id': 'role2'}, cue, 'Chinese', 930862, 'natural')

        self.assertEqual(selected[:5], result)
        self.assertEqual(selected[5]['generation_strategy'], 'clone-tilde-comma')
        self.assertEqual(selected[5]['performance_strategy'], 'restrained-boundary')

    def test_clone_performance_fallback_reaches_model_generate_with_distinct_options(self):
        class RecordingModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return [SimpleNamespace(
                    audio=np.ones(24000, dtype=np.float32) * .1,
                    sample_rate=24000,
                    token_count=12,
                )]

        with tempfile.NamedTemporaryFile(suffix='.wav') as reference:
            model = RecordingModel()
            voice = {
                'mode': 'clone',
                'clone_consent': True,
                'ref_audio': reference.name,
                'ref_text': '确认参考文本',
                'reference_revision': 'rev-1',
                'ref_text_confirmed': True,
                'ref_text_confirmed_revision': 'rev-1',
            }
            for cue_id, emotion in enumerate((
                '自然、清晰、克制，句首语气词只发一次',
                '轻快、含着笑，句首两个“嘿”清晰发音',
            ), 1):
                w._generate_audio_candidate(
                    model,
                    'clone',
                    voice,
                    {'id': cue_id, 'text': '测试。', 'emotion': emotion},
                    'Chinese',
                    930862,
                    'natural',
                )

        self.assertEqual(
            [(call['temperature'], call['top_k'], call['top_p']) for call in model.calls],
            [(0.66, 30, 0.86), (0.78, 42, 0.92)],
        )
        self.assertEqual([call['ref_text'] for call in model.calls], ['确认参考文本', '确认参考文本'])

    def test_clone_pronunciation_retry_returns_verified_candidate(self):
        cues = [
            {
                'id': 38,
                'role': 'role3',
                'text': '嘿嘿！放心吧，我明白！',
                'emotion': '轻快、含着笑',
            },
            {
                'id': 73,
                'role': 'role3',
                'text': '啊！阿芙，它飞下来了！',
                'emotion': '激动、语气上扬、重点用力',
            },
        ]
        result = (np.ones(24000, dtype=np.float32) * .1, 24000, {'temperature': .72}, None, 12)
        for cue in cues:
            seen = []

            def candidate(*args):
                seen.append(dict(args[3]))
                return result

            def transcript(*_):
                if seen[-1].get('_performance_strategy'):
                    return cue['text']
                return '放心吧我明白' if cue['id'] == 38 else '阿福他飞下来了'

            with patch.object(w, '_generate_audio_candidate', side_effect=candidate), \
                    patch.object(w, 'transcribe_full_cue', side_effect=transcript):
                selected = w.generate_audio(
                    None, 'clone', {'id': 'role3'}, cue, 'Chinese', 930862, 'natural',
                )
            self.assertEqual(cue['emotion'], selected[5]['original_emotion'])
            self.assertNotEqual(cue['emotion'], selected[5]['generation_emotion'])
            self.assertEqual(selected[5]['fallback_policy'], w.CLONE_FALLBACK_POLICY)
            self.assertTrue(selected[5]['performance_strategy'])

    def test_design_pronunciation_variants_cover_verified_inputs(self):
        name_variants = dict(w.design_pronunciation_variants(
            '杰杰，快，融合末影人卡片！', 'Chinese'))
        self.assertEqual(name_variants['leading-name-杰-comma'], '杰，杰，快，融合末影人卡片！')
        self.assertEqual(name_variants['leading-name-洁-period'], '洁，洁。快，融合末影人卡片！')
        cry_variants = dict(w.design_pronunciation_variants(
            '救命...呜呜呜...我好害怕！', 'Chinese'))
        self.assertEqual(cry_variants['onomatopoeia-expressive'], '救命！乌乌乌！我好害怕！')
        laugh_variants = dict(w.design_pronunciation_variants(
            '哈哈哈，你们一个都别想跑。哈哈哈', 'Chinese'))
        self.assertEqual(
            laugh_variants['expressive-repeat-commas'],
            '哈，哈，哈，你们一个都别想跑。哈，哈，哈',
        )
        embedded_variants = dict(w.design_pronunciation_variants(
            '不许散开哦，要像我，杰杰还有杰妹一样，永远在一起！', 'Chinese'))
        self.assertEqual(
            embedded_variants['embedded-name-截-companion'],
            '不许散开哦，要像我，截，截还有截妹一样，永远在一起！',
        )
        cue88_variants = dict(w.design_pronunciation_variants(
            '就是你了！末影人卡片的高机动性！', 'Chinese'))
        self.assertEqual(
            cue88_variants['minecraft-term-boundary'],
            '就是你了。末影人卡片的高机动性！',
        )
        cue68_variants = dict(w.design_pronunciation_variants(
            '杰杰，我来帮你！', 'Chinese'))
        self.assertEqual(
            cue68_variants['support-name-截-clear'],
            '截，截。我来帮你！',
        )
        self.assertEqual(
            w.design_pronunciation_performance_variants(
                'onomatopoeia-expressive', '救命！乌乌乌！我好害怕！',
            ),
            [('controlled-fear', '惊慌、声音发紧')],
        )
        self.assertEqual(
            w.design_pronunciation_performance_variants(
                'leading-name-洁-period', '洁，洁。我来帮你！',
            ),
            [('clear-name-emphasis', '激动、语气上扬、重点用力')],
        )
        self.assertEqual(
            w.design_pronunciation_performance_variants(
                'support-name-截-clear', '截，截。我来帮你！',
            ),
            [('clear-support-name', '自然、清晰、平稳，不要拖长音')],
        )
        self.assertEqual(
            w.design_pronunciation_performance_variants(
                'leading-name-洁-period', '洁，洁。快变身，我们一起打败他！',
            ),
            [],
        )

    def test_pronunciation_performance_retry_preserves_original_emotion(self):
        cue = {
            'id': 24,
            'role': 'r',
            'text': '救命...呜呜呜...我好害怕！',
            'emotion': '恐惧升级，语速极快，带着哭腔的尖叫',
        }
        result = (np.ones(24000, dtype=np.float32) * .1, 24000, {'temperature': .7}, None, 12)
        seen = []

        def candidate(*args):
            seen.append(dict(args[3]))
            return result

        def transcript(*_):
            if seen[-1].get('_performance_strategy') == 'controlled-fear':
                return cue['text']
            return '救命我好害怕'

        with patch.object(w, '_generate_audio_candidate', side_effect=candidate), \
                patch.object(w, 'transcribe_full_cue', side_effect=transcript):
            selected = w.generate_audio(None, 'design', {}, cue, 'Chinese', 930894, 'natural')
        self.assertEqual(cue['emotion'], '恐惧升级，语速极快，带着哭腔的尖叫')
        self.assertTrue(any(
            item.get('_generation_strategy') == 'onomatopoeia-expressive'
            and item.get('_performance_strategy') is None
            for item in seen
        ))
        self.assertEqual(selected[5]['generation_strategy'], 'onomatopoeia-expressive')
        self.assertEqual(selected[5]['performance_strategy'], 'controlled-fear')
        self.assertEqual(selected[5]['generation_emotion'], '惊慌、声音发紧')

    def test_pronunciation_retry_keeps_original_cue_and_returns_metadata(self):
        cue = {'id': 19, 'role': 'r', 'text': '杰杰从来不会这么慌张！'}
        result = (np.ones(24000, dtype=np.float32) * .1, 24000, {'temperature': .7}, None, 12)
        seen = []

        def candidate(*args):
            seen.append(dict(args[3]))
            return result

        def transcript(*_):
            return cue['text'] if seen[-1].get('_generation_strategy') else '姐姐从来不会这么慌张'

        with patch.object(w, '_generate_audio_candidate', side_effect=candidate), \
                patch.object(w, 'transcribe_full_cue', side_effect=transcript):
            selected = w.generate_audio(None, 'design', {}, cue, 'Chinese', 934261, 'natural')
        self.assertEqual(cue, {'id': 19, 'role': 'r', 'text': '杰杰从来不会这么慌张！'})
        self.assertEqual(selected[:5], result)
        self.assertEqual(selected[5]['generation_strategy'], 'leading-name-杰-comma')
        self.assertEqual(selected[5]['generation_text'], '杰，杰，从来不会这么慌张！')
        self.assertEqual(selected[5]['generation_seed'], 934261)

    def test_support_name_performance_retry_returns_verified_candidate(self):
        cue = {
            'id': 68,
            'role': 'role5',
            'text': '杰杰，我来帮你！',
            'emotion': '急切支援，语速快而清晰，语气充满信任与伙伴的温情',
        }
        result = (np.ones(24000, dtype=np.float32) * .1, 24000, {'temperature': .7}, None, 12)
        seen = []

        def candidate(*args):
            seen.append(dict(args[3]))
            return result

        def transcript(*_):
            return cue['text'] if seen[-1].get('_performance_strategy') == 'clear-support-name' else '姐姐我来帮你'

        with patch.object(w, '_generate_audio_candidate', side_effect=candidate), \
                patch.object(w, 'transcribe_full_cue', side_effect=transcript):
            selected = w.generate_audio(None, 'design', {}, cue, 'Chinese', 930894, 'natural')
        self.assertEqual(cue['emotion'], '急切支援，语速快而清晰，语气充满信任与伙伴的温情')
        self.assertEqual(selected[5]['generation_strategy'], 'support-name-截-clear')
        self.assertEqual(selected[5]['generation_text'], '截，截。我来帮你！')
        self.assertEqual(selected[5]['performance_strategy'], 'clear-support-name')
        self.assertEqual(selected[5]['generation_emotion'], '自然、清晰、平稳，不要拖长音')
        self.assertTrue(any(
            item.get('_generation_strategy') == 'support-name-截-clear'
            and item.get('_performance_strategy') == 'clear-support-name'
            for item in seen
        ))

    def test_load_and_generate_calls_retry_owner_once_and_records_strategy(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / 'x.wav'
            meta = Path(d) / 'x.json'
            cue = {'id': 19, 'role': 'r', 'text': '杰杰从来不会这么慌张！'}
            strategy = {
                'generation_strategy': 'leading-name-杰-comma',
                'generation_text': '杰，杰，从来不会这么慌张！',
                'generation_seed': 934261,
                'fallback_policy': w.DESIGN_FALLBACK_POLICY,
            }
            stale_failure = raw.parent / '019-old-role.failed.json'
            stale_failure.write_text('{}')
            selected = (np.ones(24000, dtype=np.float32) * .1, 24000, {}, None, 12, strategy)
            with patch.object(w, 'load_model', return_value=object()), \
                    patch.object(w, 'generate_audio', return_value=selected) as generate, \
                    patch.object(w, 'transcribe_full_cue', return_value=cue['text']):
                w.load_and_generate(
                    'test', 'design', [(cue, {}, raw, meta, 'fingerprint', {})],
                    'Chinese', {'design': 'test'}, 'natural',
                )
            self.assertEqual(generate.call_count, 1)
            self.assertFalse(stale_failure.exists())
            saved = json.loads(meta.read_text())
            self.assertEqual(saved['generation_strategy'], strategy['generation_strategy'])
            self.assertEqual(saved['generation_text'], strategy['generation_text'])
            self.assertEqual(saved['generation_seed'], strategy['generation_seed'])
            self.assertTrue(saved['asr_quality']['passed'])

    def test_asr_unavailable_does_not_publish(self):
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'x.wav'; meta=Path(d)/'x.json'; cue=dict(id=1,role='r',text='我站在旁边。')
            with patch.object(w,'load_model',return_value=object()),patch.object(w,'generate_audio',return_value=(np.zeros(24000,dtype=np.float32),24000,{},None,12)),patch.object(w,'transcribe_audio_array',return_value=None):
                w.load_and_generate('test','design',[(cue,{},raw,meta,'test',{})],'Chinese',{'design':'test'},'natural')
            self.assertFalse(raw.exists()); self.assertFalse(meta.exists())
            failure = raw.parent / '001-r.failed.json'
            self.assertTrue(failure.exists())
            self.assertIn('ASR未返回结果', json.loads(failure.read_text())['error'])

    def test_file_asr_cache_rejects_legacy_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / 'audio.wav'
            audio.write_bytes(b'audio')
            cue = {'id': 1, 'text': '你好'}
            identity = w.file_asr_cache_key(audio, cue)
            quality = w.cue_asr_quality('你好', '你好')
            metadata = {
                'file_asr': {
                    **identity,
                    'schema': w.FILE_ASR_CACHE_SCHEMA - 1,
                    'passed': True,
                    'quality': quality,
                    'transcript': '你好',
                },
            }
            self.assertIsNone(w.cached_file_asr(metadata, audio, cue, identity))

    def test_file_asr_cache_hit_seeds_same_run_deduplication(self):
        with tempfile.TemporaryDirectory() as d:
            clean = Path(d) / 'clean.wav'
            fit = Path(d) / 'fit.wav'
            clean.write_bytes(b'identical-waveform')
            fit.write_bytes(clean.read_bytes())
            cue = {'id': 1, 'text': '你好'}
            shared = {}
            identity = w.file_asr_cache_key(clean, cue)
            record = {
                'schema': w.FILE_ASR_CACHE_SCHEMA,
                **identity,
                'passed': True,
                'quality': w.cue_asr_quality('你好', '你好'),
                'transcript': '你好',
            }
            with patch.object(w, '_validate_audio_file_record', return_value=record) as validate:
                first, first_status = w.validate_or_reuse_audio_file(clean, cue, {}, shared)
                second, second_status = w.validate_or_reuse_audio_file(fit, cue, {}, shared)
            self.assertEqual(first_status, 'checked')
            self.assertEqual(second_status, 'dedup')
            self.assertEqual(first, second)
            validate.assert_called_once()

    def test_adopted_take_cache_key_changes_when_waveform_changes(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'take.wav'
            take = {'id': 'take-1', 'sha256': 'stale-metadata'}
            source.write_bytes(b'first-waveform')
            first = w.adopted_raw_cache_key('generation', take, source)
            source.write_bytes(b'second-waveform')
            second = w.adopted_raw_cache_key('generation', take, source)
            self.assertNotEqual(first, second)

    def test_generation_failure_still_records_generation_timing(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / 'x.wav'
            meta = Path(d) / 'x.json'
            cue = {'id': 1, 'role': 'r', 'text': '你好'}
            timings = {'model_load_seconds': 0.0, 'generation_seconds': 0.0, 'generation_qa_seconds': 0.0}
            with patch.object(w, 'load_model', return_value=object()), \
                    patch.object(w, 'generate_audio', side_effect=RuntimeError('generation failed')):
                w.load_and_generate(
                    'test', 'design', [(cue, {}, raw, meta, 'test', {})],
                    'Chinese', {'design': 'test'}, 'natural', timings,
                )
            self.assertGreaterEqual(timings['generation_seconds'], 0.0)
            self.assertTrue((raw.parent / '001-r.failed.json').exists())

    def test_a_graded_take_is_published_instead_of_a_failure_marker(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / 'x.wav'
            meta = Path(d) / 'x.json'
            cue = dict(id=1, role='r', text='杰杰还在养病。')
            take = (np.full(2400, .1, dtype=np.float32), 24000, {}, None, 12,
                    {'best_effort': True, 'degraded': False, 'gate_label': 'ordinary:930001',
                     'gate_transcript': '姐姐还在养病', 'gate_quality': {'passed': False}})
            with patch.object(w, 'load_model', return_value=object()), \
                    patch.object(w, 'generate_audio', return_value=take), \
                    patch.object(w, 'transcribe_full_cue', return_value='姐姐还在养病'), \
                    patch.object(w, 'cue_asr_quality', return_value={
                        'passed': False, 'length_ok': True, 'phonetic_ok': False,
                        'final_ok': True, 'similarity': 0.7, 'match_mode': 'rejected'}):
                w.load_and_generate('test', 'clone', [(cue, {}, raw, meta, 'test', {})],
                                    'Chinese', {'clone': 'test'}, 'natural')

            self.assertTrue(raw.is_file())
            self.assertTrue(meta.is_file())
            self.assertFalse((raw.parent / '001-r.failed.json').exists())
            saved = json.loads(meta.read_text())
            self.assertFalse(saved['asr_quality']['passed'])
            self.assertTrue(saved['best_effort'])
            self.assertTrue(saved['needs_review'])
            self.assertEqual(saved['gate_transcript'], '姐姐还在养病')
            self.assertTrue(w.raw_cache_approved(saved))

    def test_a_cue_with_no_usable_take_still_records_a_failure(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / 'x.wav'
            meta = Path(d) / 'x.json'
            cue = dict(id=1, role='r', text='杰杰还在养病。')
            with patch.object(w, 'load_model', return_value=object()), \
                    patch.object(w, 'generate_audio', side_effect=w.CandidateRejected('时长异常')):
                w.load_and_generate('test', 'clone', [(cue, {}, raw, meta, 'test', {})],
                                    'Chinese', {'clone': 'test'}, 'natural')

            self.assertFalse(raw.exists())
            self.assertFalse(meta.exists())
            self.assertTrue((raw.parent / '001-r.failed.json').is_file())

    def test_atomic_json_writer_leaves_valid_json_and_no_temp_file(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'record.json'
            w.write_json_atomic(target, {'ok': True, 'text': '中文'})
            self.assertEqual(json.loads(target.read_text()), {'ok': True, 'text': '中文'})
            self.assertEqual(list(target.parent.glob('.record.json.*.tmp')), [])


class ReviewedTakeTests(unittest.TestCase):
    """A line the recognizer cannot transcribe must be ruled on by a person, not a looser gate."""

    def candidate(self, value, seed):
        return (np.full(2400, value, dtype=np.float32), 24000, {}, None, 12, {'generation_seed': seed})

    def rejected(self, length_ok, similarity):
        return {
            'passed': False, 'length_ok': length_ok, 'phonetic_ok': False,
            'final_ok': length_ok, 'similarity': similarity, 'match_mode': 'rejected',
        }

    def test_best_effort_sink_returns_the_least_bad_take_instead_of_failing(self):
        cue = {'id': 8, 'role': 'role1', 'text': '杰杰还在养病。', 'emotion': '自然'}
        calls = {'n': 0}

        def take(*_args, **_kwargs):
            calls['n'] += 1
            return self.candidate(.1 * calls['n'], calls['n'])

        def transcript(*_args, **_kwargs):
            return f'识别{calls["n"]}'

        def quality(*_args, **_kwargs):
            # Every later candidate is a better match, so the sink must keep the last
            # one rather than the first take it happened to see.
            return self.rejected(True, 0.5 + calls['n'] / 100)

        sink = {}
        with patch.object(w, '_generate_audio_candidate', side_effect=take) as generate, \
                patch.object(w, 'transcribe_full_cue', side_effect=transcript), \
                patch.object(w, 'cue_asr_quality', side_effect=quality), \
                patch.object(w, 'clone_pronunciation_variants', return_value=[]):
            result = w.generate_audio(None, 'clone', {'id': 'role1'}, cue, 'Chinese', 1, review_sink=sink)

        self.assertTrue(result[5]['best_effort'])
        self.assertEqual(result[5]['gate_transcript'], f'识别{generate.call_count}')
        self.assertEqual(result[5]['gate_quality']['similarity'], 0.5 + generate.call_count / 100)
        self.assertEqual(sink['best']['quality']['similarity'], 0.5 + generate.call_count / 100)
        np.testing.assert_allclose(result[0], np.full(2400, .1 * generate.call_count, dtype=np.float32))
        self.assertGreater(generate.call_count, 1)

    def test_fast_review_run_skips_prosody_only_sweeps(self):
        cue = {'id': 98, 'role': 'role1', 'text': '杰妹，是你救了我吗？谢谢你。', 'emotion': '自然'}
        with patch.object(w, '_generate_audio_candidate', return_value=self.candidate(.1, 1)) as generate, \
                patch.object(w, 'transcribe_full_cue', return_value='姐妹是你救了我吗谢谢你'), \
                patch.object(w, 'cue_asr_quality', return_value=self.rejected(False, 0.8)), \
                patch.object(w, 'clone_pronunciation_variants', return_value=[]):
            w.generate_audio(None, 'clone', {'id': 'role1'}, cue, 'Chinese', 1,
                             review_sink={}, fast_review=True)
            review_generations = generate.call_count
        with patch.object(w, '_generate_audio_candidate', return_value=self.candidate(.1, 1)) as generate, \
                patch.object(w, 'transcribe_full_cue', return_value='姐妹是你救了我吗谢谢你'), \
                patch.object(w, 'cue_asr_quality', return_value=self.rejected(False, 0.8)), \
                patch.object(w, 'clone_pronunciation_variants', return_value=[]):
            with self.assertRaises(RuntimeError):
                w.generate_audio(None, 'clone', {'id': 'role1'}, cue, 'Chinese', 1)

        self.assertLess(review_generations, generate.call_count)

    def test_the_project_build_keeps_the_best_take_instead_of_failing(self):
        cue = {'id': 8, 'role': 'role1', 'text': '杰杰还在养病。', 'emotion': '自然'}
        with patch.object(w, '_generate_audio_candidate', return_value=self.candidate(.1, 1)), \
                patch.object(w, 'transcribe_full_cue', return_value='姐姐还在养病'), \
                patch.object(w, 'cue_asr_quality', return_value=self.rejected(True, 0.9)), \
                patch.object(w, 'clone_pronunciation_variants', return_value=[]):
            result = w.generate_audio(None, 'clone', {'id': 'role1'}, cue, 'Chinese', 1, review_sink={})
        self.assertTrue(result[5]['best_effort'])
        self.assertEqual(result[5]['gate_transcript'], '姐姐还在养病')

    def test_a_runaway_take_is_never_offered_as_best_effort(self):
        # 20s of audio for a line allowed 3s is runaway generation, not a slow read.
        take = (np.full(24000 * 20, .1, dtype=np.float32), 24000, {}, {'max_seconds': 3.0, 'max_tokens': 100}, 90)
        exc = w.CandidateRejected('克隆结果时长异常', take=take, degraded=True)
        self.assertIsNone(w.offerable_take(exc))
        bounded = (np.full(24000 * 6, .1, dtype=np.float32), 24000, {}, {'max_seconds': 3.0, 'max_tokens': 100}, 40)
        self.assertIsNotNone(w.offerable_take(w.CandidateRejected('时长超出', take=bounded, degraded=True)))

    def test_a_graded_take_is_reused_instead_of_rolled_again(self):
        metadata = {'asr_quality': {'passed': False}, 'best_effort': True}
        self.assertTrue(w.raw_cache_approved(metadata))
        self.assertFalse(w.raw_cache_approved({'asr_quality': {'passed': False}}))
        self.assertTrue(w.raw_cache_approved({'asr_quality': {'passed': True}}))

    def test_without_a_sink_generation_still_fails_closed(self):
        cue = {'id': 8, 'role': 'role1', 'text': '杰杰还在养病。', 'emotion': '自然'}
        with patch.object(w, '_generate_audio_candidate', return_value=self.candidate(.1, 1)), \
                patch.object(w, 'transcribe_full_cue', return_value='姐姐还在养病'), \
                patch.object(w, 'cue_asr_quality', return_value=self.rejected(True, 0.9)), \
                patch.object(w, 'clone_pronunciation_variants', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, '没有产出可用音频'):
                w.generate_audio(None, 'clone', {'id': 'role1'}, cue, 'Chinese', 1)

    def _take(self, root, reviewed):
        takes = root / 'takes'
        takes.mkdir(parents=True, exist_ok=True)
        path = takes / 'cue-008-take.wav'
        wavfile.write(path, 24000, np.full(2400, .1, dtype=np.float32))
        take = {'id': 'take-1', 'file': path.name, 'sha256': w.file_sha256(path)}
        if reviewed:
            take.update({
                'reviewed': True, 'reviewed_at': '2026-09-17T15:00:00',
                'review_transcript': '姐姐还在养病',
                'review_expected_text': '杰杰还在养病。',
                'review_gate_reason': 'clone-pronunciation:clone-name-pair-杰',
            })
        return take

    def test_a_signed_off_take_is_reused_without_rerunning_the_gate(self):
        cue = {'id': 8, 'transcript': '杰杰还在养病。'}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cue['adopted_take'] = self._take(root, reviewed=True)
            cue['text'] = '杰杰还在养病。'
            with patch.object(w, 'transcribe_full_cue', return_value='姐姐还在养病'), \
                    patch.object(w, 'cue_asr_quality', return_value=self.rejected(True, 0.9)) as gate:
                source, audit, _ = w.adopted_take_source(root, cue, include_validation=True)
            gate.assert_not_called()
            self.assertTrue(source.is_file())
            self.assertEqual(audit['review_transcript'], '姐姐还在养病')

    def test_an_unsigned_take_the_gate_rejects_is_refused(self):
        cue = {'id': 8, 'text': '杰杰还在养病。'}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cue['adopted_take'] = self._take(root, reviewed=False)
            with patch.object(w, 'transcribe_full_cue', return_value='姐姐还在养病'), \
                    patch.object(w, 'cue_asr_quality', return_value=self.rejected(True, 0.9)):
                with self.assertRaisesRegex(RuntimeError, '完整性校验失败'):
                    w.adopted_take_source(root, cue, include_validation=True)

    def test_reviewed_take_audit_requires_the_human_record_to_be_intact(self):
        base = {
            'reviewed': True, 'reviewed_at': '2026-09-17T15:00:00',
            'review_transcript': '姐姐还在养病', 'review_expected_text': '杰杰还在养病。',
        }
        self.assertIsNotNone(w.reviewed_take_audit(base))
        self.assertIsNone(w.reviewed_take_audit({**base, 'reviewed_at': ''}))
        self.assertIsNone(w.reviewed_take_audit({**base, 'review_transcript': '  '}))
        self.assertIsNone(w.reviewed_take_audit({**base, 'reviewed': False}))

if __name__=='__main__': unittest.main()
